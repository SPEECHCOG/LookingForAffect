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
from collections import Counter

from utils import extract_df_from_trankit


def get_utt_syntax_tree_height(FA_dict):
    
    distances = []
    
    for token in FA_dict["tokens"]:
        
        token_distance = 0
        
        if pd.isna(token):
            feats.append(np.nan)
            continue
        
        if "head" in token.keys():
            
            head = token["head"]
            
            while head != 0 or token_distance == 100:
                head = FA_dict["tokens"][head-1]["head"]
                token_distance = token_distance+1
            
            distances.append(token_distance)
            
        else:
            
            distances.append(np.nan)
            
    
    return np.max(distances)+1


def parse_feats(feats):
    if feats in (None, "_", ""):
        return {}
    return dict(f.split("=") for f in feats.split("|"))
    


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")

    args = parser.parse_args()

    # -------------------------------------------------
    # Load trankit tagging results
    # -------------------------------------------------
    coll = False
    
    if coll:
    
        trankit_tags_file = args.data_dir+"/"+"trankit_coll.pkl"

        
    else: 
        trankit_tags_file = args.data_dir+"/"+"trankit_non_coll.pkl"

        

    
    
    with open(trankit_tags_file, "rb") as f:
        trankit_tags = pickle.load(f)
    
    trankit_FA_dicts = extract_df_from_trankit(trankit_tags)
    
    FA_ids = []
    utt_feats = []
    utt_syntax_tree_len = []
    
    
    
    #Parse utterance features from trankit dictionaries,
    #compute syntax tree height, and collect feats, upos, xpos and deprel values
    for FA_dict in trankit_FA_dicts:
        
        FA_ids.append(FA_dict["utt_id"])
        feats = []
        FA_utt_id = FA_dict["utt_id"]
        print(FA_dict["utt_id"])
        
        if FA_utt_id == 45:
            print("")
        
        if pd.isna(FA_dict["text"]):
            feats.append(np.nan)
            utt_feats.append((FA_utt_id, feats))
            utt_syntax_tree_len.append(0)
            continue
        
        utt_syntax_tree_len.append(get_utt_syntax_tree_height(FA_dict))
        
        for token in FA_dict["tokens"]:
            
            if pd.isna(token):
                feats.append(np.nan)
                continue
            
            if "feats" in token.keys():
            
                features = parse_feats(token["feats"])
                
            features["upos"] = token["upos"]
            features["xpos"] = token["xpos"]
            features["deprel"] = token["deprel"]
                
            feats.append(features)

            
        utt_feats.append((FA_utt_id, feats))
    
    

    #collect the feature vobabulary
    #i.e. each individual feature = value combination there is 
    feat_vocab = Counter()
    
    for i, (FA_utt_id, utt) in enumerate(utt_feats):
        for word_feats in utt:
            
            if pd.isna(word_feats):
                continue
            
            for k, v in word_feats.items():
                feat_vocab[(k, v)] += 1
    
    # fix column order
    feat_list = sorted(feat_vocab)
    
    columns = [f"{k}={v}" for k, v in feat_list]
    columns.append("utt_len")
    columns.append("utt_id")
    
    
    rows = []
    #compute the occurences of each feature = value combination
    #for each utterance and store to rows
    for i, (FA_utt_id, utt) in enumerate(utt_feats):
        counts = Counter()

        for word_feats in utt:
            if pd.isna(word_feats):
                continue
            for k, v in word_feats.items():
                counts[f"{k}={v}"] += 1
        
        counts["utt_len"] = sum(1 for word_feats in utt if not pd.isna(word_feats))
        counts["utt_id"] = FA_ids[i]
        rows.append(counts)
        
    df = pd.DataFrame(rows, columns=columns)
    df = df.fillna(0).astype(int)
    
    feat_cols = df.columns.difference(["utt_len"])
    df_norm = df.copy()
    df_norm[feat_cols] = df_norm[feat_cols].div(df_norm["utt_len"], axis=0)
    df_norm[feat_cols] = df_norm[feat_cols].fillna(0)
    
    #read multiindex, df["Case", "Gen"] 
    #df.columns = pd.MultiIndex.from_tuples(c.split("=") for c in df.columns)
    #df_norm.columns = pd.MultiIndex.from_tuples(c.split("=") for c in df_norm.columns)
    
    df["utt_id"] = FA_ids
    df_norm["utt_id"] = FA_ids
    
    df["syntax_tree_len"] = utt_syntax_tree_len
    df_norm["syntax_tree_len"] = utt_syntax_tree_len
    
    if coll:
        
        df.to_csv(args.data_dir+"/"+"trankit_coll_feats.csv")
        df_norm.to_csv(args.data_dir+"/"+"trankit_coll_normalized_feats.csv")
        
    else:
        
        df.to_csv(args.data_dir+"/"+"trankit_noncoll_feats.csv")
        df_norm.to_csv(args.data_dir+"/"+"trankit_noncoll_normalized_feats.csv")
        
    