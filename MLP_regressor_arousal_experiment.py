#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arousal regression with feature fusion and CCC loss
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
import os

import random

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
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--modernbert", type=str, help="ModernBERT embeddings pkl", default="./data/FinmodernBERT_noncoll_embeddings.pkl")
    parser.add_argument("--prosody", type=str, help="Prosodic features csv", default = "./data/all_JP_features_df.csv")
    parser.add_argument("--egemaps", type=str, help="Egemaps features pkl", default = "./data/eGeMAPS_features.pkl")
    parser.add_argument("--exhubert", type=str, help="ExHuBERT features pkl", default="./data/ExHuBERT_embeddings.pkl")
    parser.add_argument("--sentiment", type=str, help="FinnSentiment posteriors pkl", default="./data/Finnsentiment_coll_posteriors.pkl")
    parser.add_argument("--SELF", type=str, help="SELF based features pkl", default="./data/SELF_noncoll_features.pkl")
    parser.add_argument("--FEIL", type=str, help="SELF based features pkl", default="./data/FEIL_noncoll_features.pkl")
    parser.add_argument("--Affnorms210", type=str, help="SELF based features pkl", default="./data/Affnorms_210_noncoll_features.pkl")
    parser.add_argument("--Affnorms420", type=str, help="SELF based features pkl", default="./data/Affnorms_420_noncoll_features.pkl")
    parser.add_argument("--lexicon", type=str, help="Lexicon based features pkl", default="./data/lexicon_noncoll_features.pkl")
    
    parser.add_argument("--ling", type=str, help="Linguistic features csv", default = "./data/trankit_noncoll_feats.csv")
    parser.add_argument("--lingnorm", type=str, help="Linguistic features csv", default = "./data/trankit_noncoll_normalized_feats.csv")
    
    
    
    parser.add_argument("--metadata", type=str, default="/Path/To/FinnAffect/root/annotations_and_metadata/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    # feature toggles
    parser.add_argument("--use_bert", action="store_true", default=False)
    parser.add_argument("--use_prosody", action="store_true", default=False)
    parser.add_argument("--use_egemaps", action="store_true", default=False)
    parser.add_argument("--use_exhubert", action="store_true", default=False)
    parser.add_argument("--use_sentiment", action="store_true", default=False)
    parser.add_argument("--use_valence", action="store_true", default=False)
    parser.add_argument("--use_self", action="store_true", default=False)
    parser.add_argument("--use_feil", action="store_true", default=False)
    parser.add_argument("--use_affnorms210", action="store_true", default=False)
    parser.add_argument("--use_affnorms420", action="store_true", default=False)
    parser.add_argument("--use_lexicon", action="store_true", default=False)
    parser.add_argument("--use_ling_norm", action="store_true", default=False)
    parser.add_argument("--use_ling", action="store_true", default=False)
    parser.add_argument("--coll", action="store_true", default=False)
    parser.add_argument("--noncoll", action="store_true", default=False)
    
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.coll:
        coll = True
        
    if args.noncoll:
        
        coll = False

    # -------------------------------------------------
    # Load labels
    # -------------------------------------------------
    annotators = ["a1", "a2", "a3", "a4", "a5"]
    valence_df = pd.read_csv("./data/valence_normalized.csv")
    arousal_df = pd.read_csv("./data/arousal_normalized.csv")
    metadata_df = pd.read_csv(args.metadata+"metadata.csv")
    
    annotated_ids = arousal_df[annotators].dropna(how="all").index
    arousal_annotated_df = arousal_df.loc[annotated_ids]
    
    arousal_GS_df = arousal_annotated_df.dropna(how="any")
    arousal_single_annotation_df = arousal_annotated_df.drop(arousal_GS_df.index)
    
    GS_meta_df = metadata_df.loc[arousal_GS_df.index]
    single_annotation_meta_df = metadata_df.loc[arousal_single_annotation_df.index]
    
    speakers_GS = set(GS_meta_df["speaker id"])
    
    tr_val_df = arousal_single_annotation_df.loc[
        ~single_annotation_meta_df["speaker id"].isin(speakers_GS)]["mean"]
    
    tr_val_speakers = single_annotation_meta_df["speaker id"].loc[tr_val_df.index]
    
    GS_test_df = arousal_GS_df["mean"]
    a1_test_df = arousal_GS_df["a1"]
    a2_test_df = arousal_GS_df["a2"]
    a3_test_df = arousal_GS_df["a3"]
    a4_test_df = arousal_GS_df["a4"]
    a5_test_df = arousal_GS_df["a5"]
    
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
                                             "Reflex",
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
                                             "Reflex",
                                             "Voice", "VerbForm", "Tense", 
                                             "Polarity", "Degree", "Mood", 
                                             "Person", "upos", "xpos", "syntax_tree_len"]]

        feature_dfs.append(ling_norm_features_df.astype(float))
        
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
        
        test_set_name = test_set_name+"_"+"ExHuBERT"
        
    if args.use_egemaps:
        print("Using egemaps features")
        feature_dfs.append(load_feature_pkl(args.egemaps, normalize = True))
        
        test_set_name = test_set_name+"_"+"eGeMAPS"

    if args.use_sentiment:
        print("Using FinnSentiment posteriors")
        feature_dfs.append(load_feature_pkl(args.sentiment))
        
        test_set_name = test_set_name+"_"+"FinnSentiment"
        
    if args.use_valence:
        print("Using valence annotations")
        valence_as_feature_df = valence_df.loc[annotated_ids]
        feature_dfs.append(valence_as_feature_df["mean"])
        
        test_set_name = test_set_name+"_"+"Valence"

    if not feature_dfs:
        raise ValueError("No features selected!")

    X_df = pd.concat(feature_dfs, axis=1, join="inner")
    X_tr_val = X_df.loc[tr_val_ids].values
    X_tst = X_df.loc[tst_ids].values

    print(f"Final feature dim: {X_tr_val.shape[1]}")

    # -------------------------------------------------
    # Group K-Fold setup
    # -------------------------------------------------
    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    groups = tr_val_speakers.values  # speaker IDs as strings or ints
    
    fold_val_cccs = []
    fold_losses = []
    fold_test_cccs = {"GS": [], "a1": [], "a2": [], "a3": [], "a4": [], "a5": []}
    fold_best_epochs = []
    
    # -------------------------------------------------
    # CV loop
    # -------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tr_val, y_tr_val, groups)):
        
        fold_seed = args.seed + fold
        set_seed(fold_seed)
        
        print(f"\n=== Fold {fold+1}/{n_folds} ===")
    
        X_train, X_val = X_tr_val[train_idx], X_tr_val[val_idx]
        y_train, y_val = y_tr_val[train_idx], y_tr_val[val_idx]
    
        train_ds = FeatureDataset(X_train, y_train)
        val_ds = FeatureDataset(X_val, y_val)
    
        g = torch.Generator()
        g.manual_seed(fold_seed)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g
            )
        
        
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
        # -------------------------------------------------
        # Model, loss, optimizer
        # -------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        model = MLPRegressor(input_dim=X_tr_val.shape[1]).to(device)
        criterion = CCCLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
        best_val_ccc = -np.inf
        best_state = None
        best_epoch = None
        val_cccs = []
        tr_losses = []
    
        # -------------------------------------------------
        # Training loop
        # -------------------------------------------------
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
    
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
    
            # validation
            model.eval()
            with torch.no_grad():
                preds, targets = [], []
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds.append(model(xb).cpu())
                    targets.append(yb.cpu())
    
                preds = torch.cat(preds)
                targets = torch.cat(targets)
                val_ccc = 1.0 - criterion(preds, targets).item()
    
            # save best model state for this fold
            if val_ccc > best_val_ccc:
                best_val_ccc = val_ccc
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
    
            val_cccs.append(val_ccc)
            tr_losses.append(train_losses)
    
            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {np.mean(train_losses):.4f} | "
                f"Val CCC: {val_ccc:.4f}"
            )
    
        # -------------------------------------------------
        # Load best model for fold
        # -------------------------------------------------
        model.load_state_dict(best_state)
    
        # store fold validation CCC
        fold_val_cccs.append(val_cccs)
        
        fold_losses.append(tr_losses)
        
        fold_best_epochs.append(best_epoch)
    
        # -------------------------------------------------
        # Evaluate on test set (GS) if desired
        # -------------------------------------------------
        for testset_name, data in test_datasets.items():
            
            testset_y, testset_ids = data
            
            test_ds = FeatureDataset(X_tst, testset_y)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        
            model.eval()
            with torch.no_grad():
                test_preds, test_targets = [], []
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    test_preds.append(model(xb).cpu())
                    test_targets.append(yb.cpu())
        
                test_preds = torch.cat(test_preds)
                test_targets = torch.cat(test_targets)
                test_ccc = 1.0 - criterion(test_preds, test_targets).item()
        
            fold_test_cccs[testset_name].append(test_ccc)
            print(f"Fold {fold+1} {testset_name} Test CCC: {test_ccc:.4f}, Best Epoch: {best_epoch}")
    
    # -------------------------------------------------
    # Aggregate CV results
    # -------------------------------------------------
    print("\n=== Cross-Validation Summary ===")
    print(f"Validation CCC: mean {np.mean(fold_val_cccs):.4f}, std {np.std(fold_val_cccs):.4f}")
    print(f"Best epoch mean {np.mean(fold_best_epochs):.4f}")
    

    results = {
        "Set name": test_set_name,
        "Fold Val CCCs": fold_val_cccs,
        "Fold Test CCCs": fold_test_cccs,
        "Fold train losses": fold_losses,
        "Fold best epochs": fold_best_epochs,
        "Batch size": args.batch_size,
        "Epochs": args.epochs,
        "Learning rate": args.lr,
        "coll": coll,
        "initial seed": args.seed
    }
    
    
    if coll:
    
        results_dir = args.data_dir+"/results_coll_arousal_"+str(args.seed)+"/"
            
    else: 
        
        results_dir = args.data_dir+"/results_noncoll_arousal_"+str(args.seed)+"/"
        
    # Optionally save aggregated results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        
    torch.save(model.state_dict(), results_dir+"best_model_"+test_set_name+".pt")    
    
    with open(results_dir+"cv_results"+test_set_name+".pkl", 'wb') as f:
        pickle.dump(results, f)
    
    
    
    
    
    
    
    
    
    