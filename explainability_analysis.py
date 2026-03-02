#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Valence regression with feature fusion and CCC loss
Author: Kalle Lahtinen
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.inspection import permutation_importance
import os

import random
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------
# CCC loss
# -------------------------------------------------

class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        
        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)
        
        pred = pred.squeeze()
        target = target.squeeze()

        mean_pred = torch.mean(pred)
        mean_tgt = torch.mean(target)

        var_pred = torch.var(pred, unbiased=False)
        var_tgt = torch.var(target, unbiased=False)

        cov = torch.mean((pred - mean_pred) * (target - mean_tgt))

        ccc = (2 * cov) / (
            var_pred + var_tgt + (mean_pred - mean_tgt) ** 2 + self.eps
        )
        return 1.0 - ccc

def my_custom_scorer(y_true, y_pred):
    criterion = CCCLoss()
    
    return criterion(y_pred, y_tst).item()
    
# -------------------------------------------------
# Dataset
# -------------------------------------------------

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------------------------------
# MLP regressor
# -------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        
        h1 = min(512, max(128, input_dim // 2))
        h2 = min(256, h1)
        h3 = min(128, h2)
        
        hidden_dims = (h1, h2, h3)

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    
    
class PyTorchRegressorWrapper():
    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model
    
    def fit(self, X, y=None):
        # For permutation importance, we don't actually need to retrain
        # Just return self to satisfy the interface
        return self
    
    def predict(self, X):
        # Convert to tensor and predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            if hasattr(self.pytorch_model, 'net'):  # Your custom model structure
                predictions = self.pytorch_model.net(X_tensor).numpy()
            else:
                predictions = self.pytorch_model(X_tensor).numpy()
        return predictions.flatten()
    


# -------------------------------------------------
# Utility: load feature pickles
# -------------------------------------------------

def load_feature_pkl(path, normalize = False):
    
    columns = None
    
    with open(path, "rb") as f:
        
        feature_obj = pickle.load(f)
        
        if len(feature_obj) == 3:
            
            ids, feats, columns = feature_obj
            
        else:
            
            ids, feats = feature_obj
        
    if normalize: 
        F = feats

        dims = np.shape(F)
        
        F_normalized = np.zeros(dims)
        
        for dim in range(dims[1]):
            
            f = F[:,dim]
            
            f_u = np.mean(f)
            
            f_std = np.std(f)
            
            f_norm = (f - f_u) / f_std
            
            F_normalized[:,dim] = f_norm
        
        feats = F_normalized
        
    if columns: 
            
        return pd.DataFrame(feats, index=ids, columns=columns).astype(float)
    
    else: 

        return pd.DataFrame(feats, index=ids).astype(float)
# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    
    
    #TODO: Tsekkaa tallennettavat metriikat, että on kaikki tarpeellinen
    #Toteuta annotaattorispesifi testi GS:n lisäksi

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--modernbert", type=str, help="ModernBERT embeddings pkl", default="./data/FinmodernBERT_noncoll_embeddings.pkl")
    parser.add_argument("--prosody", type=str, help="Prosodic features csv", default = "./data/all_JP_features_df.csv")
    parser.add_argument("--egemaps", type=str, help="Egemaps features pkl", default = "./data/eGeMAPS_features.pkl")
    parser.add_argument("--exhubert", type=str, help="ExHuBERT features pkl", default="./data/ExHuBERT_embeddings.pkl")
    parser.add_argument("--sentiment", type=str, help="FinnSentiment posteriors pkl", default="./data/Finnsentiment_noncoll_posteriors.pkl")
    parser.add_argument("--SELF", type=str, help="SELF based features pkl", default="./data/SELF_noncoll_features.pkl")
    parser.add_argument("--FEIL", type=str, help="SELF based features pkl", default="./data/FEIL_noncoll_features.pkl")
    parser.add_argument("--Affnorms210", type=str, help="SELF based features pkl", default="./data/Affnorms_210_noncoll_features.pkl")
    parser.add_argument("--Affnorms420", type=str, help="SELF based features pkl", default="./data/Affnorms_420_noncoll_features.pkl")
    parser.add_argument("--lexicon", type=str, help="Lexicon based features pkl", default="./data/lexicon_noncoll_features.pkl")
    
    parser.add_argument("--ling", type=str, help="Linguistic features csv", default = "./data/trankit_noncoll_feats.csv")
    parser.add_argument("--lingnorm", type=str, help="Linguistic features csv", default = "./data/trankit_noncoll_normalized_feats.csv")
    
    
    
    parser.add_argument("--metadata", type=str, default="/Volumes/T9/LP_HP_TP_combined/Master_datasets/FinnAffect_Kielipankki/annotations_and_metadata/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    # feature toggles
    parser.add_argument("--use_bert", action="store_true", default=False)
    parser.add_argument("--use_prosody", action="store_true", default=False)
    parser.add_argument("--use_egemaps", action="store_true", default=False)
    parser.add_argument("--use_exhubert", action="store_true", default=False)
    parser.add_argument("--use_sentiment", action="store_true", default=False)
    parser.add_argument("--use_arousal", action="store_true", default=False)
    parser.add_argument("--use_self", action="store_true", default=False)
    parser.add_argument("--use_feil", action="store_true", default=False)
    parser.add_argument("--use_affnorms210", action="store_true", default=False)
    parser.add_argument("--use_affnorms420", action="store_true", default=False)
    parser.add_argument("--use_lexicon", action="store_true", default=False)
    parser.add_argument("--use_ling_norm", action="store_true", default=True)
    parser.add_argument("--use_ling", action="store_true", default=False)
    parser.add_argument("--coll", action="store_true", default=False)
    parser.add_argument("--noncoll", action="store_true", default=False)
    
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.coll:
        coll = False
        
    if args.noncoll:
        
        coll = True

    # -------------------------------------------------
    # Load labels
    # -------------------------------------------------
    annotators = ["a1", "a2", "a3", "a4", "a5"]
    valence_df = pd.read_csv(args.metadata+"valence_normalized.csv")
    arousal_df = pd.read_csv(args.metadata+"arousal_normalized.csv")
    metadata_df = pd.read_csv(args.metadata+"metadata.csv")
    
    
    annotated_ids = valence_df[annotators].dropna(how="all").index
    valence_annotated_df = valence_df.loc[annotated_ids]
    
    valence_GS_df = valence_annotated_df.dropna(how="any")
    valence_single_annotation_df = valence_annotated_df.drop(valence_GS_df.index)
    
    GS_meta_df = metadata_df.loc[valence_GS_df.index]
    single_annotation_meta_df = metadata_df.loc[valence_single_annotation_df.index]
    
    
    speakers_GS = set(GS_meta_df["speaker id"])
    
    tr_val_df = valence_single_annotation_df.loc[
        ~single_annotation_meta_df["speaker id"].isin(speakers_GS)]["mean"]
    
    tr_val_speakers = single_annotation_meta_df["speaker id"].loc[tr_val_df.index]
    
    GS_test_df = valence_GS_df["mean"]
    a1_test_df = valence_GS_df["a1"]
    a2_test_df = valence_GS_df["a2"]
    a3_test_df = valence_GS_df["a3"]
    a4_test_df = valence_GS_df["a4"]
    a5_test_df = valence_GS_df["a5"]
    
    y_tr_val = tr_val_df.values
    tr_val_ids = tr_val_df.index
    
    y_tst = GS_test_df.values
    tst_ids = GS_test_df.index
    
    y_a1_tst = a1_test_df.values
    a1_tst_ids = a1_test_df.index
    
    y_a2_tst = a2_test_df.values
    a2_tst_ids = a2_test_df.index
    
    y_a3_tst = a3_test_df.values
    a3_tst_ids = a3_test_df.index
    
    y_a4_tst = a4_test_df.values
    a4_tst_ids = a4_test_df.index
    
    y_a5_tst = a5_test_df.values
    a5_tst_ids = a5_test_df.index
    
    test_datasets = {"GS": (y_tst, tst_ids),
                     "a1": (y_a1_tst, a1_tst_ids),
                     "a2": (y_a2_tst, a2_tst_ids),
                     "a3": (y_a3_tst, a3_tst_ids),
                     "a4": (y_a4_tst, a4_tst_ids),
                     "a5": (y_a5_tst, a5_tst_ids)}
    
    test_set_name = ""
    feature_names = []

    # -------------------------------------------------
    # Load and combine features
    # -------------------------------------------------

    feature_dfs = []

    if args.use_bert:
        print("Using ModernBERT embeddings")
        feature_dfs.append(load_feature_pkl(args.modernbert))
        
        test_set_name = test_set_name+"_"+"ModernBERT"
        
    if args.use_self:
        print("Using SELF based vectors")
        feature_dfs.append(load_feature_pkl(args.SELF))
        
        test_set_name = test_set_name+"_"+"SELF"
        
    if args.use_feil:
        print("Using FEIL based vectors")
        feature_dfs.append(load_feature_pkl(args.FEIL))
        
        test_set_name = test_set_name+"_"+"FEIL"
        
    if args.use_affnorms210:
        print("Using Affnorms210 based vectors")
        feature_dfs.append(load_feature_pkl(args.Affnorms210))
        
        test_set_name = test_set_name+"_"+"Affnorms210"
        
    if args.use_affnorms420:
        print("Using Affnorms420 based vectors")
        feature_dfs.append(load_feature_pkl(args.Affnorms420))
        
        test_set_name = test_set_name+"_"+"Affnorms420"
        
    if args.use_lexicon:
        print("Using combined lexicon based vectors")
        feature_dfs.append(load_feature_pkl(args.lexicon))
        
        feature_names.extend(list(feature_dfs[-1].columns))
        
        test_set_name = test_set_name+"_"+"Lexicon"

        
    if args.use_ling:
        print("Using linguistic features")
        #feature_dfs.append(load_feature_pkl(args.prosody))
        ling_features_df = pd.read_csv(args.ling)
        ling_features_df.set_index("utt_id", inplace=True)
        ling_features_df.drop(columns=["Unnamed: 0"], inplace=True)
        
        ling_features_df.columns = pd.MultiIndex.from_tuples(c.split("=") for c in ling_features_df.columns)
        
        
        ling_features_df = ling_features_df[["Case", "Clitic", "Connegative",
                                             "Derivation", "PronType",
                                             "InfForm", "NumType", "Number", "PartForm",
                                             "Reflex", "Voice",
                                             "Voice", "VerbForm", "Tense", 
                                             "Polarity", "Degree", "Mood", 
                                             "Person", "upos", "xpos", "utt_len", "syntax_tree_len"]]
        
        feature_dfs.append(ling_features_df.astype(float))
        
        test_set_name = test_set_name+"_"+"Ling"
        
    if args.use_ling_norm:
        print("Using normalized linguistic features")
        #feature_dfs.append(load_feature_pkl(args.prosody))
        ling_norm_features_df = pd.read_csv(args.lingnorm)
        ling_norm_features_df.set_index("utt_id", inplace=True)
        ling_norm_features_df.drop(columns=["Unnamed: 0"], inplace=True)
        
        
        ling_norm_features_df.columns = pd.MultiIndex.from_tuples(c.split("=") for c in ling_norm_features_df.columns)
        
        ling_norm_features_df = ling_norm_features_df[["Case", "Clitic", "Connegative",
                                             "Derivation", "PronType",
                                             "InfForm", "NumType", "Number", "PartForm",
                                             "Reflex", "Voice",
                                             "Voice", "VerbForm", "Tense", 
                                             "Polarity", "Degree", "Mood", 
                                             "Person", "upos", "xpos", "syntax_tree_len"]]
        

        

        feature_dfs.append(ling_norm_features_df.astype(float))
        
        feature_names.extend(list(feature_dfs[-1].columns))
        
        test_set_name = test_set_name+"_"+"Lingnorm"
        
    if args.use_prosody:
        print("Using prosodic features")
        #feature_dfs.append(load_feature_pkl(args.prosody))
        JP_features_df = pd.read_csv(args.prosody)
        JP_features_df.set_index("utt_id", inplace=True)
        JP_features_df.fillna(0.0, inplace=True)
        JP_features_df.drop(columns=['speaker_gender'], inplace=True)
        
        for column in JP_features_df.columns:
            JP_features_df[column] = (JP_features_df[column] - JP_features_df[column].mean()) / JP_features_df[column].std()
        
        feature_dfs.append(JP_features_df.astype(float))
        
        test_set_name = test_set_name+"_"+"Prosody"
        
    if args.use_exhubert:
        print("Using exhubert features")
        feature_dfs.append(load_feature_pkl(args.exhubert))
        
        feature_names.extend(list(feature_dfs[-1].columns))
        
        test_set_name = test_set_name+"_"+"ExHuBERT"
        
    if args.use_egemaps:
        print("Using egemaps features")
        feature_dfs.append(load_feature_pkl(args.egemaps, normalize = True))
        
        test_set_name = test_set_name+"_"+"eGeMAPS"

    if args.use_sentiment:
        print("Using FinnSentiment posteriors")
        feature_dfs.append(load_feature_pkl(args.sentiment))
        
        test_set_name = test_set_name+"_"+"FinnSentiment"
        
    if args.use_arousal:
        print("Using Arousal annotations")
        arousal_as_feature_df = arousal_df.loc[annotated_ids]
        feature_dfs.append(arousal_as_feature_df["mean"])
        
        test_set_name = test_set_name+"_"+"Arousal"

    if not feature_dfs:
        raise ValueError("No features selected!")

    X_df = pd.concat(feature_dfs, axis=1, join="inner")
    X_tr_val = X_df.loc[tr_val_ids].values
    X_tst = X_df.loc[tst_ids].values
    criterion = CCCLoss()

    print(f"Final feature dim: {X_tr_val.shape[1]}")
    
    
    ##XGBOOS test
    model = xgb.XGBRegressor(objective='reg:squarederror',
                         n_estimators=128, random_state=args.seed)
    
    model.fit(X_tr_val, y_tr_val)
    y_pred = model.predict(X_tst)
    
    rmse = np.sqrt(mean_squared_error(y_tst, y_pred))
    r2 = r2_score(y_tst, y_pred)
    CCC_score = 1.0 - criterion(y_pred, y_tst).item()
    
    print(f'RMSE: {rmse:.3f}')
    print(f'R²: {r2:.3f}')
    print(f'CCC: {CCC_score:.3f}')
    
    plot_importance(model, importance_type="gain")
    plt.show()
    
    
    ##Feature importance permutation
    
    
    score = make_scorer(my_custom_scorer, greater_is_better=False)
    
    valence = True
    
    data_root = "/Users/lahtine9/workwork/python_ml_utils/LookingForValence/data/results_all_2/"
    
    if valence:
    
        coll_data = data_root+"/results_coll_valence_42/"
        non_coll_data = data_root+"/results_noncoll_valence_42/"
        
    else:
        coll_data = data_root+"/results_coll_arousal_42/"
        non_coll_data = data_root+"/results_noncoll_arousal_42/"
        
    
    coll_model_path = coll_data+"best_model_"+test_set_name+".pt"
    non_coll_model_path = non_coll_data+"best_model_"+test_set_name+".pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    coll_model = MLPRegressor(input_dim=X_tr_val.shape[1]).to(device)
    coll_model.load_state_dict(torch.load(coll_model_path, weights_only=True, map_location=torch.device('cpu')))
    coll_model = PyTorchRegressorWrapper(coll_model)
    
    non_coll_model = MLPRegressor(input_dim=X_tr_val.shape[1]).to(device)
    non_coll_model.load_state_dict(torch.load(coll_model_path, weights_only=True, map_location=torch.device('cpu')))
    non_coll_model = PyTorchRegressorWrapper(non_coll_model)
    
    
    r = permutation_importance(coll_model, X_tst, y_tst, n_repeats=30, random_state=args.seed, scoring=score)

    print("Coll importances: ")
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            #print(f"{feature_names[i]:<8} "
            #      f"{r.importances_mean[i]:.3f} "
            #      f"+/- {r.importances_std[i]:.3f}")
            print(feature_names[i])
            print(str(r.importances_mean[i]))
            print("+-"+str(r.importances_std[i]))
            
    r = permutation_importance(non_coll_model, X_tst, y_tst, n_repeats=30, random_state=args.seed, scoring=score)

    print("Non Coll importances: ")
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            #print(f"{feature_names[i]:<8} "
            #      f"{r.importances_mean[i]:.3f} "
            #      f"+/- {r.importances_std[i]:.3f}")
            print(feature_names[i])
            print(str(r.importances_mean[i]))
            print("+-"+str(r.importances_std[i]))
    
    