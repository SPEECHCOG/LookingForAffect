#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:19:59 2025

@author: lahtine9
"""

import numpy as np
import sys
import os
import argparse
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle
from scipy import stats

#TODO: tsekkaa valence gap etc paprut

SMALLER_SIZE = 8
SMALL_SIZE = 15
MEDIUM_SIZE = 20
MEDIUM_LARGE_SIZE = 25
BIG_SIZE = 35
BIGGER_SIZE = 50
width = 24
height = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})





AUDIO_FEATURES = {"ExHuBERT", "eGeMAPS"}
TEXT_FEATURES  = {"ModernBERT", "Lexicon", "Lingnorm", "FinnSentiment"}
ANNOTATION_FEATURES = {"Arousal", "Valence"}   # to exclude entirely


def parse_features(name: str):
    """Split an index entry into component feature names."""
    return set(name.split("_"))


def filter_audio_only(df: pd.DataFrame):
    """
    Keep rows that have audio features only (no text features),
    and exclude rows containing Arousal/Valence.
    """
    mask = []
    for idx in df.index:
        feats = parse_features(idx)
        has_audio = len(feats & AUDIO_FEATURES) > 0
        has_text = len(feats & TEXT_FEATURES) > 0
        has_annotations = len(feats & ANNOTATION_FEATURES) > 0

        mask.append(has_audio and not has_text and not has_annotations)

    return df[mask]


def filter_text_only(df: pd.DataFrame):
    """
    Keep rows that have text features only (no audio features),
    and exclude rows containing Arousal/Valence.
    """
    mask = []
    for idx in df.index:
        feats = parse_features(idx)
        has_audio = len(feats & AUDIO_FEATURES) > 0
        has_text = len(feats & TEXT_FEATURES) > 0
        has_annotations = len(feats & ANNOTATION_FEATURES) > 0

        mask.append(has_text and not has_audio and not has_annotations)

    return df[mask]


    

if __name__ == '__main__':
    
    
    valence = True
    
    data_root = "./data/results_all/"
    
    
    if valence:
    
        coll_data = data_root+"/results_coll_valence_42/"
        non_coll_data = data_root+"/results_noncoll_valence_42/"
        
    else:
        coll_data = data_root+"/results_coll_arousal_42/"
        non_coll_data = data_root+"/results_noncoll_arousal_42/"
        
    
    noncoll_tr_losses = {}
    noncoll_val_CCCs = {}
    GS_noncoll_test_CCCs = {}
    a1_noncoll_test_CCCs = {}
    a2_noncoll_test_CCCs = {}
    a3_noncoll_test_CCCs = {}
    a4_noncoll_test_CCCs = {}
    a5_noncoll_test_CCCs = {}
    
    GS_noncoll_fold_test_CCCs = {}
    
    noncoll_val_mean_CCCs = {}
    
    for fname in os.listdir(non_coll_data):
        if fname.endswith(".pkl"):
            pkl_path = os.path.join(non_coll_data, fname)
            print(f"Processing {pkl_path}")
            
            with open(pkl_path, "rb") as f:
                results = pickle.load(f)
                
                name = results["Set name"][1:]
                
                if results["coll"] == True:
                    print("wyt")
                
                best_fold_validations = []
                
                for i, best_val_epoch in enumerate(results["Fold best epochs"]):
                    
                    best_fold_validations.append(results["Fold Val CCCs"][i][best_val_epoch-1])
                
                noncoll_val_mean_CCCs[name] = (np.mean(best_fold_validations), np.std(best_fold_validations))
                GS_noncoll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["GS"]), np.std(results["Fold Test CCCs"]["GS"]))
                a1_noncoll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a1"]), np.std(results["Fold Test CCCs"]["a1"]))
                a2_noncoll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a2"]), np.std(results["Fold Test CCCs"]["a2"]))
                a3_noncoll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a3"]), np.std(results["Fold Test CCCs"]["a3"]))
                a4_noncoll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a4"]), np.std(results["Fold Test CCCs"]["a4"]))
                a5_noncoll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a5"]), np.std(results["Fold Test CCCs"]["a5"]))
                
                
                for j, fold_test_CCC in enumerate(results["Fold Test CCCs"]["GS"]):
                    GS_noncoll_fold_test_CCCs[name+"_"+str(j)] = fold_test_CCC
                
                
                
                
    
    
    GS_noncoll_fold_test_CCCs_df = pd.DataFrame.from_dict(GS_noncoll_fold_test_CCCs, orient="index")
    GS_noncoll_fold_test_CCCs_df.columns = ["GS Noncoll mean CCC"]
    
    
    GS_noncoll_test_df = pd.DataFrame.from_dict(GS_noncoll_test_CCCs, orient="index")
    GS_noncoll_test_df.columns = ["GS Noncoll mean CCC", "GS Noncoll CCC std"]
    
    a1_noncoll_test_df = pd.DataFrame.from_dict(a1_noncoll_test_CCCs, orient="index")
    a1_noncoll_test_df.columns = ["a1 Noncoll mean CCC", "a1 Noncoll CCC std"]
    
    a2_noncoll_test_df = pd.DataFrame.from_dict(a2_noncoll_test_CCCs, orient="index")
    a2_noncoll_test_df.columns = ["a2 Noncoll mean CCC", "a2 Noncoll CCC std"]
    
    a3_noncoll_test_df = pd.DataFrame.from_dict(a3_noncoll_test_CCCs, orient="index")
    a3_noncoll_test_df.columns = ["a3 Noncoll mean CCC", "a3 Noncoll CCC std"]
    
    a4_noncoll_test_df = pd.DataFrame.from_dict(a4_noncoll_test_CCCs, orient="index")
    a4_noncoll_test_df.columns = ["a4 Noncoll mean CCC", "a4 Noncoll CCC std"]
    
    a5_noncoll_test_df = pd.DataFrame.from_dict(a5_noncoll_test_CCCs, orient="index")
    a5_noncoll_test_df.columns = ["a5 Noncoll mean CCC", "a5 Noncoll CCC std"]
    
    
        
    noncoll_val_df = pd.DataFrame.from_dict(noncoll_val_mean_CCCs, orient="index")
    noncoll_val_df.columns = ["Noncoll mean validation CCC", "Noncoll CCC validation std"]
    
    
    coll_tr_losses = {}
    coll_val_CCCs = {}
    GS_coll_test_CCCs = {}
    a1_coll_test_CCCs = {}
    a2_coll_test_CCCs = {}
    a3_coll_test_CCCs = {}
    a4_coll_test_CCCs = {}
    a5_coll_test_CCCs = {}
    
    GS_coll_fold_test_CCCs = {}
    
    coll_val_mean_CCCs = {}
    
    for fname in os.listdir(coll_data):
        if fname.endswith(".pkl"):
            pkl_path = os.path.join(coll_data, fname)
            print(f"Processing {pkl_path}")
            
            with open(pkl_path, "rb") as f:
                results = pickle.load(f)
                
                name = results["Set name"][1:]
                
                best_fold_validations = []
                
                for i, best_val_epoch in enumerate(results["Fold best epochs"]):
                    
                    best_fold_validations.append(results["Fold Val CCCs"][i][best_val_epoch-1])
                
                coll_val_mean_CCCs[name] = (np.mean(best_fold_validations), np.std(best_fold_validations))
                GS_coll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["GS"]), np.std(results["Fold Test CCCs"]["GS"]))
                a1_coll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a1"]), np.std(results["Fold Test CCCs"]["a1"]))
                a2_coll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a2"]), np.std(results["Fold Test CCCs"]["a2"]))
                a3_coll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a3"]), np.std(results["Fold Test CCCs"]["a3"]))
                a4_coll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a4"]), np.std(results["Fold Test CCCs"]["a4"]))
                a5_coll_test_CCCs[name] = (np.mean(results["Fold Test CCCs"]["a5"]), np.std(results["Fold Test CCCs"]["a5"]))
                
                
                for j, fold_test_CCC in enumerate(results["Fold Test CCCs"]["GS"]):
                    GS_coll_fold_test_CCCs[name+"_"+str(j)] = fold_test_CCC
    
    
    
    GS_coll_fold_test_CCCs_df = pd.DataFrame.from_dict(GS_coll_fold_test_CCCs, orient="index")
    GS_coll_fold_test_CCCs_df.columns = ["GS coll mean CCC"]
    
    GS_coll_test_df = pd.DataFrame.from_dict(GS_coll_test_CCCs, orient="index")
    GS_coll_test_df.columns = ["GS coll mean CCC", "GS coll CCC std"]
    
    a1_coll_test_df = pd.DataFrame.from_dict(a1_coll_test_CCCs, orient="index")
    a1_coll_test_df.columns = ["a1 coll mean CCC", "a1 coll CCC std"]
    
    a2_coll_test_df = pd.DataFrame.from_dict(a2_coll_test_CCCs, orient="index")
    a2_coll_test_df.columns = ["a2 coll mean CCC", "a2 coll CCC std"]
    
    a3_coll_test_df = pd.DataFrame.from_dict(a3_coll_test_CCCs, orient="index")
    a3_coll_test_df.columns = ["a3 coll mean CCC", "a3 coll CCC std"]
    
    a4_coll_test_df = pd.DataFrame.from_dict(a4_coll_test_CCCs, orient="index")
    a4_coll_test_df.columns = ["a4 coll mean CCC", "a4 coll CCC std"]
    
    a5_coll_test_df = pd.DataFrame.from_dict(a5_coll_test_CCCs, orient="index")
    a5_coll_test_df.columns = ["a5 coll mean CCC", "a5 coll CCC std"]
    
    
        
    coll_val_df = pd.DataFrame.from_dict(coll_val_mean_CCCs, orient="index")
    coll_val_df.columns = ["coll mean validation CCC", "coll CCC validation std"]
    


    combined_df = pd.concat([GS_coll_test_df, GS_noncoll_test_df, a1_coll_test_df, a1_noncoll_test_df, a2_coll_test_df, a2_noncoll_test_df, a3_coll_test_df, a3_noncoll_test_df, a4_coll_test_df, a4_noncoll_test_df, a5_coll_test_df, a5_noncoll_test_df], axis=1)
    
    combined_test_val_df = pd.concat([combined_df, coll_val_df,  noncoll_val_df], axis=1)
    
    annotator_coll_scores = ["a1 coll mean CCC", "a2 coll mean CCC", "a3 coll mean CCC", "a4 coll mean CCC", "a5 coll mean CCC"]
    annotator_noncoll_scores = ["a1 Noncoll mean CCC", "a2 Noncoll mean CCC", "a3 Noncoll mean CCC", "a4 Noncoll mean CCC", "a5 Noncoll mean CCC"]
    
    
    combined_test_val_df["coll annotator std"] = combined_test_val_df[annotator_coll_scores].std(axis=1)
    
    combined_test_val_df["noncoll annotator std"] = combined_test_val_df[annotator_noncoll_scores].std(axis=1)
    
    combined_test_val_df = combined_test_val_df.round(3)
    
    combined_test_val_df["coll-noncoll-GS"] = combined_test_val_df["GS coll mean CCC"] - combined_test_val_df["GS Noncoll mean CCC"]
    
    ann_score_effect_coll = {}
    ann_score_effect_noncoll = {}
    
    for feature_comb_name in combined_test_val_df.index:
        
        if valence:
            annotation_feature = "_Arousal"
        else:
            annotation_feature = "_Valence"
            
        if annotation_feature in feature_comb_name:
            
            annotation_feature_wo_annotation = feature_comb_name.split(annotation_feature)[0]
            
            feature_score_coll = combined_test_val_df.loc[feature_comb_name]['GS coll mean CCC']
            feature_score_noncoll = combined_test_val_df.loc[feature_comb_name]['GS Noncoll mean CCC']
            
            feature_score_wo_annotation_coll = combined_test_val_df.loc[annotation_feature_wo_annotation]['GS coll mean CCC']
            feature_score_wo_annotation_noncoll = combined_test_val_df.loc[annotation_feature_wo_annotation]['GS Noncoll mean CCC']
            
            
            ann_score_effect_coll[feature_comb_name] = [feature_score_coll - feature_score_wo_annotation_coll, feature_score_coll, feature_score_wo_annotation_coll]
            
            
            ann_score_effect_noncoll[feature_comb_name] = [feature_score_noncoll - feature_score_wo_annotation_noncoll, feature_score_noncoll, feature_score_wo_annotation_noncoll]
        
        else:
            
            ann_score_effect_coll[feature_comb_name] = [np.nan, np.nan, np.nan]
            ann_score_effect_noncoll[feature_comb_name] = [np.nan, np.nan, np.nan]
        
    ann_score_effect_coll_df = pd.DataFrame.from_dict(ann_score_effect_coll, orient="index", columns=["Ann. as feature effect coll", "Feature score with annotation", "Feature score wo annotation"])
    ann_score_effect_noncoll_df = pd.DataFrame.from_dict(ann_score_effect_noncoll, orient="index", columns=["Ann. as feature effect noncoll", "Feature score with annotation", "Feature score wo annotation"])
    
    combined_test_val_df = combined_test_val_df.join(ann_score_effect_coll_df["Ann. as feature effect coll"])
    combined_test_val_df = combined_test_val_df.join(ann_score_effect_noncoll_df["Ann. as feature effect noncoll"])
    
    combined_test_val_audio_df = filter_audio_only(combined_test_val_df)
    combined_test_val_text_df = filter_text_only(combined_test_val_df)
    
    GS_noncoll_fold_test_CCCs_text_df = filter_text_only(GS_noncoll_fold_test_CCCs_df)
    GS_coll_fold_test_CCCs_text_df = filter_text_only(GS_noncoll_fold_test_CCCs_df)
    
    CF_SF_text_only = (combined_test_val_text_df["coll-noncoll-GS"].mean().round(3), np.round(combined_test_val_text_df["coll-noncoll-GS"].std(),3))
    
    annotation_imporvement_CF = (combined_test_val_df['Ann. as feature effect coll'].mean().round(3), np.round(combined_test_val_df['Ann. as feature effect coll'].std(),3))
    annotation_imporvement_SF = (combined_test_val_df['Ann. as feature effect noncoll'].mean().round(3), np.round(combined_test_val_df['Ann. as feature effect noncoll'].std(),3))
    
    
    CCC_correlations = combined_test_val_df[["GS coll mean CCC", 
                                             "a1 coll mean CCC",
                                             "a2 coll mean CCC",
                                             "a3 coll mean CCC",
                                             "a4 coll mean CCC",
                                             "a5 coll mean CCC",
                                             "GS Noncoll mean CCC", 
                                             "a1 Noncoll mean CCC",
                                             "a2 Noncoll mean CCC",
                                             "a3 Noncoll mean CCC",
                                             "a4 Noncoll mean CCC",
                                             "a5 Noncoll mean CCC",
                                             "coll mean validation CCC",
                                             "Noncoll mean validation CCC"]].corr()
    
    CCC_correlations.rename({"GS coll mean CCC": "GS coll",
                             "a1 coll mean CCC": "a1 coll",
                             "a2 coll mean CCC": "a2 coll",
                             "a3 coll mean CCC": "a3 coll",
                             "a4 coll mean CCC": "a4 coll",
                             "a5 coll mean CCC": "a5 coll",
                             "GS Noncoll mean CCC": "GS noncoll", 
                             "a1 Noncoll mean CCC": "a1 noncoll",
                             "a2 Noncoll mean CCC": "a2 noncoll",
                             "a3 Noncoll mean CCC": "a3 noncoll",
                             "a4 Noncoll mean CCC": "a4 noncoll",
                             "a5 Noncoll mean CCC": "a5 noncoll",
                             "coll mean validation CCC": "coll val",
                             "Noncoll mean validation CCC": "noncoll val"}, inplace=True)
    
    
    
    noncoll_top5 = combined_test_val_df.nlargest(3, "Noncoll mean validation CCC")
    
    if not valence:
    
        noncoll_top5 = pd.concat([noncoll_top5, combined_test_val_df.loc[["ModernBERT_Lexicon_Lingnorm_FinnSentiment","ExHuBERT_eGeMAPS", "ModernBERT", "Lexicon", "Lingnorm", "FinnSentiment", "eGeMAPS", "ExHuBERT", "Valence"]]])
    
    else: 
        
        noncoll_top5 = pd.concat([noncoll_top5, combined_test_val_df.loc[["ModernBERT_Lexicon_Lingnorm_FinnSentiment","ExHuBERT_eGeMAPS", "ModernBERT", "Lexicon", "Lingnorm", "FinnSentiment", "eGeMAPS", "ExHuBERT", "Arousal"]]])
    
        
    noncoll_top5_combined = noncoll_top5.apply(lambda row: (row['GS coll mean CCC'], row['GS coll CCC std']), axis=1).to_frame(name="GS coll")
    noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['GS Noncoll mean CCC'], row['GS Noncoll CCC std']), axis=1).to_frame(name="GS Noncoll")], axis=1)
    
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a1 Noncoll mean CCC'], row['a1 Noncoll CCC std']), axis=1).to_frame(name="a1 Noncoll")], axis=1)
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a1 coll mean CCC'], row['a1 coll CCC std']), axis=1).to_frame(name="a1 coll")], axis=1)
    
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a2 Noncoll mean CCC'], row['a2 Noncoll CCC std']), axis=1).to_frame(name="a2 Noncoll")], axis=1)
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a2 coll mean CCC'], row['a2 coll CCC std']), axis=1).to_frame(name="a2 coll")], axis=1)
    
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a3 Noncoll mean CCC'], row['a3 Noncoll CCC std']), axis=1).to_frame(name="a3 Noncoll")], axis=1)
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a3 coll mean CCC'], row['a3 coll CCC std']), axis=1).to_frame(name="a3 coll")], axis=1)
    
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a4 Noncoll mean CCC'], row['a4 Noncoll CCC std']), axis=1).to_frame(name="a4 Noncoll")], axis=1)
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a4 coll mean CCC'], row['a4 coll CCC std']), axis=1).to_frame(name="a4 coll")], axis=1)
    
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a5 Noncoll mean CCC'], row['a5 Noncoll CCC std']), axis=1).to_frame(name="a5 Noncoll")], axis=1)
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['a5 coll mean CCC'], row['a5 coll CCC std']), axis=1).to_frame(name="a5 coll")], axis=1)
    
    noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['Noncoll mean validation CCC'], row['Noncoll CCC validation std']), axis=1).to_frame(name="Validation Noncoll")], axis=1)
    #noncoll_top5_combined = pd.concat([noncoll_top5_combined, noncoll_top5.apply(lambda row: (row['coll mean validation CCC'], row['coll CCC validation std']), axis=1).to_frame(name="Validation coll")], axis=1)
    
    noncoll_top5_combined = noncoll_top5_combined.apply(lambda col: col.map(lambda x: (float(x[0]), float(x[1]))))
    
    noncoll_top5_combined = noncoll_top5_combined.join(combined_test_val_df[["coll annotator std", "noncoll annotator std", "coll-noncoll-GS", "Ann. as feature effect coll", "Ann. as feature effect noncoll"]], how="left")
    
    coll_top5 = combined_test_val_df.nlargest(3, "coll mean validation CCC")
    
    if not valence:
    
        coll_top5 = pd.concat([coll_top5, combined_test_val_df.loc[["ModernBERT_Lexicon_Lingnorm_FinnSentiment","ExHuBERT_eGeMAPS", "ModernBERT", "Lexicon", "Lingnorm", "FinnSentiment", "eGeMAPS", "ExHuBERT", "Valence"]]])
    
    else: 
        
        coll_top5 = pd.concat([coll_top5, combined_test_val_df.loc[["ModernBERT_Lexicon_Lingnorm_FinnSentiment","ExHuBERT_eGeMAPS","ModernBERT", "Lexicon", "Lingnorm", "FinnSentiment", "eGeMAPS", "ExHuBERT", "Arousal"]]])
    
    
    coll_top5_combined = coll_top5.apply(lambda row: (row['GS coll mean CCC'], row['GS coll CCC std']), axis=1).to_frame(name="GS coll")
    coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['GS Noncoll mean CCC'], row['GS Noncoll CCC std']), axis=1).to_frame(name="GS Noncoll")], axis=1)
    
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a1 Noncoll mean CCC'], row['a1 Noncoll CCC std']), axis=1).to_frame(name="a1 Noncoll")], axis=1)
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a1 coll mean CCC'], row['a1 coll CCC std']), axis=1).to_frame(name="a1 coll")], axis=1)
    
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a2 Noncoll mean CCC'], row['a2 Noncoll CCC std']), axis=1).to_frame(name="a2 Noncoll")], axis=1)
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a2 coll mean CCC'], row['a2 coll CCC std']), axis=1).to_frame(name="a2 coll")], axis=1)
    
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a3 Noncoll mean CCC'], row['a3 Noncoll CCC std']), axis=1).to_frame(name="a3 Noncoll")], axis=1)
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a3 coll mean CCC'], row['a3 coll CCC std']), axis=1).to_frame(name="a3 coll")], axis=1)
    
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a4 Noncoll mean CCC'], row['a4 Noncoll CCC std']), axis=1).to_frame(name="a4 Noncoll")], axis=1)
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a4 coll mean CCC'], row['a4 coll CCC std']), axis=1).to_frame(name="a4 coll")], axis=1)
    
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a5 Noncoll mean CCC'], row['a5 Noncoll CCC std']), axis=1).to_frame(name="a5 Noncoll")], axis=1)
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['a5 coll mean CCC'], row['a5 coll CCC std']), axis=1).to_frame(name="a5 coll")], axis=1)
    
    #coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['Noncoll mean validation CCC'], row['Noncoll CCC validation std']), axis=1).to_frame(name="Validation Noncoll")], axis=1)
    coll_top5_combined = pd.concat([coll_top5_combined, coll_top5.apply(lambda row: (row['coll mean validation CCC'], row['coll CCC validation std']), axis=1).to_frame(name="Validation coll")], axis=1)
    
    coll_top5_combined = coll_top5_combined.apply(lambda col: col.map(lambda x: (float(x[0]), float(x[1]))))
    coll_top5_combined = coll_top5_combined.join(combined_test_val_df[["coll annotator std", "noncoll annotator std","coll-noncoll-GS", "Ann. as feature effect coll", "Ann. as feature effect noncoll"]], how="left")
    
    
    
    
    coll_top5_combined_forcomparison = coll_top5_combined.copy()
    coll_top5_combined_forcomparison.index = coll_top5_combined_forcomparison.index.map(lambda x: f"{x} | coll")
    
    noncoll_top5_combined_forcomparison = noncoll_top5_combined.copy()
    noncoll_top5_combined_forcomparison.index = noncoll_top5_combined_forcomparison.index.map(lambda x: f"{x} | noncoll")
    
    combined_comparison = pd.concat([coll_top5_combined_forcomparison, noncoll_top5_combined_forcomparison], axis=0)
    
    
    
    if valence:
    
        noncoll_top5_combined.to_csv(non_coll_data+"noncoll_top5_val.csv")
        coll_top5_combined.to_csv(coll_data+"coll_top5_val.csv")
        
        combined_test_val_df.to_csv(data_root+"all_results_val.csv")
        
        combined_test_val_df["coll-noncoll-GS"].to_csv(data_root+"coll_noncoll_diff_val.csv")
        
    else:
        noncoll_top5_combined.to_csv(non_coll_data+"noncoll_top5_ar.csv")
        coll_top5_combined.to_csv(coll_data+"coll_top5_ar.csv")
        
        combined_test_val_df.to_csv(data_root+"all_results_ar.csv")
        
        combined_test_val_df["coll-noncoll-GS"].to_csv(data_root+"coll_noncoll_diff_ar.csv")
        
        
    #pairwise t-test for text only CF and SF CCC-scores
    CF_SF_textonly_ttest = stats.ttest_rel(combined_test_val_text_df["GS coll mean CCC"], combined_test_val_text_df["GS Noncoll mean CCC"])
    CF_SF_audioonly_ttest = stats.ttest_rel(combined_test_val_audio_df["GS coll mean CCC"], combined_test_val_audio_df["GS Noncoll mean CCC"])
    CF_SF_audio_and_text_ttest = stats.ttest_rel(combined_test_val_df["GS coll mean CCC"], combined_test_val_df["GS Noncoll mean CCC"])
    
    
    ann_score_effect_ttest_coll = stats.ttest_rel(ann_score_effect_coll_df["Feature score with annotation"].dropna(), ann_score_effect_coll_df["Feature score wo annotation"].dropna())
    ann_score_effect_ttest_noncoll = stats.ttest_rel(ann_score_effect_noncoll_df["Feature score with annotation"].dropna(), ann_score_effect_noncoll_df["Feature score wo annotation"].dropna())
    
    
    
    all_features = combined_test_val_df.index.to_list()
    folds = 5 
    
    CF_SF_diff_fold_ttest_results = {}
    CF_SF_diff_fold_ttest_results_for_reported = {}
    
    if valence:        
        reported_features = ["ModernBERT_ExHuBERT_FinnSentiment_Arousal", "ModernBERT_Lexicon_Lingnorm_ExHuBERT_FinnSentiment_Arousal", "ModernBERT_Lexicon_ExHuBERT_FinnSentiment_Arousal", "ModernBERT_Lexicon_Lingnorm_FinnSentiment","ExHuBERT_eGeMAPS","ModernBERT", "Lexicon", "Lingnorm", "FinnSentiment", "eGeMAPS", "ExHuBERT", "Arousal"]
    
    else:
        reported_features = ["ExHuBERT_eGeMAPS_FinnSentiment_Valence", "ModernBERT_Lexicon_ExHuBERT_eGeMAPS_FinnSentiment_Valence", "Lexicon_Lingnorm_ExHuBERT_eGeMAPS_Valence", "ModernBERT_Lexicon_Lingnorm_FinnSentiment","ExHuBERT_eGeMAPS","ModernBERT", "Lexicon", "Lingnorm", "FinnSentiment", "eGeMAPS", "ExHuBERT", "Valence"]
    
    
    for feature in all_features:
        
        
        feature_folds = list(map(lambda i: f"{feature}_{i}", range(folds)))
        
        feature_noncoll_folds = GS_noncoll_fold_test_CCCs_df.loc[GS_noncoll_fold_test_CCCs_df.index.isin(feature_folds)] 
        feature_coll_folds = GS_coll_fold_test_CCCs_df.loc[GS_coll_fold_test_CCCs_df.index.isin(feature_folds)] 
        
        
        stat, p_value = stats.ttest_rel(feature_coll_folds["GS coll mean CCC"], feature_noncoll_folds["GS Noncoll mean CCC"])
        
        if p_value < 0.10:
        
            CF_SF_diff_fold_ttest_results[feature] = (p_value, np.mean(feature_coll_folds["GS coll mean CCC"]-feature_noncoll_folds["GS Noncoll mean CCC"]))
            
            if feature in reported_features:
                CF_SF_diff_fold_ttest_results_for_reported[feature] = (p_value, np.mean(feature_coll_folds["GS coll mean CCC"]-feature_noncoll_folds["GS Noncoll mean CCC"]))
            
        
        else: 
            
            CF_SF_diff_fold_ttest_results[feature] = ("-", np.mean(feature_coll_folds["GS coll mean CCC"]-feature_noncoll_folds["GS Noncoll mean CCC"]))
            
            if feature in reported_features:
                CF_SF_diff_fold_ttest_results_for_reported[feature] = (p_value, np.mean(feature_coll_folds["GS coll mean CCC"]-feature_noncoll_folds["GS Noncoll mean CCC"]))
            
            
            
            