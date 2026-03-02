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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from utils import extract_df_from_trankit

def add_lexicon_name_to_columns(column_list, name):
    
    new_columns = []
    
    for dim_name in column_list:
        
        new_columns.append(name+"_"+dim_name)
        
    return new_columns

    
def get_lexicon_vectors(utt_lemmas, lexicon_df, lexicon_emo_dims):
    
    #EMO DIM eli määrittele kolumnien nimet
    #Lisää nollavektori utterancelle, jos nolla matchia
    #laske matchien määrä utterancessa ja palauta myös se
    
    lexicon_vectors = []
    coverage = []
    emo_dim = len(lexicon_emo_dims)
    
    for utterance in utt_lemmas:
        
        utt_vectors = []

        for lemma in utterance:
            key = lemma.lower()
            if key in lexicon_df.index:
                
                matches = lexicon_df.loc[key]
                
                #check if the lemma is found several times in the lexicon
                #in that case, use the mean over all occurances as the 
                #lemma vector
                if isinstance(matches, pd.DataFrame):
                    vec = matches[lexicon_emo_dims].values.mean(axis=0)
                    
                else:
                    vec = vec = matches[lexicon_emo_dims].values
                
                utt_vectors.append(vec)
                
        try: 
            if len(utt_vectors) == 0:
                utt_vec = np.zeros(emo_dim)
                coverage.append(0)
            else:
                utt_vec = np.mean(utt_vectors, axis=0)
                coverage.append(len(utt_vectors))
        except Exception as e:
            print(e)

        
        lexicon_vectors.append(utt_vec)
       
    #concatenate utterance level lexicon score vectors and the coverage count
    #i.e. how many findings per utterance from the individual lemmas
    #into one feature vector dim = emo_dims + 1
    lexicon_vectors = np.array(lexicon_vectors)   # (N, D)
    coverage = np.array(coverage).reshape(-1, 1)  # (N, 1)
    
    lexicon_vectors_with_cov = np.concatenate(
        [lexicon_vectors, coverage],
        axis=1
    )
        
    return lexicon_vectors_with_cov, list(lexicon_emo_dims)+["coverage"]


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
        SELF_target_file = args.data_dir+"/SELF_coll_features.pkl"
        FEIL_target_file = args.data_dir+"/FEIL_coll_features.pkl"
        Affnorms_210_target_file = args.data_dir+"/Affnorms_210_coll_features.pkl"
        Affnorms_420_target_file = args.data_dir+"/Affnorms_420_coll_features.pkl"
        lexicon_vectors_target_file = args.data_dir+"/lexicon_coll_features.pkl"
        
    else: 
        trankit_tags_file = args.data_dir+"/"+"trankit_non_coll.pkl"
        SELF_target_file = args.data_dir+"/SELF_noncoll_features.pkl"
        FEIL_target_file = args.data_dir+"/FEIL_noncoll_features.pkl"
        Affnorms_210_target_file = args.data_dir+"/Affnorms_210_noncoll_features.pkl"
        Affnorms_420_target_file = args.data_dir+"/Affnorms_420_noncoll_features.pkl"
        lexicon_vectors_target_file = args.data_dir+"/lexicon_noncoll_features.pkl"
        
    SELF_path = args.data_dir+"/"+"SELF-FEIL/SELF.tsv"
    FEIL_path = args.data_dir+"/"+"SELF-FEIL/FEIL.tsv"
    Affective_norms_201_nouns_path = args.data_dir+"/"+"Affektiaineisto_EilolaHavelka/Eilola & Havelka_Affective Norms for 210 British English and Finnish Nouns.xls"
    Affective_norms_420_nouns_path = args.data_dir+"/"+"Affektidata_SöderholmHäyryLaineKarrasch/Database_S1.xls"
    
    SELF_df = pd.read_csv(SELF_path, sep="\t")
    SELF_df["word"] = SELF_df["word"].str.lower().str.strip()
    SELF_df.drop(columns=["Sum of labels"], inplace=True)
    SELF_df.set_index("word", inplace=True)
    SELF_emo_dims = ["positive", "negative", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    
    #transform the "emotion" category into one hot encoded emotion vector
    #drop the english word and the emotion columns
    FEIL_df = pd.read_csv(FEIL_path, sep="\t")
    FEIL_one_hot_encoding = pd.get_dummies(FEIL_df["emotion"], dtype=int)
    FEIL_df.drop(columns=["word", "emotion"], inplace=True)
    FEIL_df = pd.concat([FEIL_df, FEIL_one_hot_encoding], axis=1)
    FEIL_df["finnish-fi"] = FEIL_df["finnish-fi"].str.lower().str.strip()
    FEIL_df.set_index("finnish-fi", inplace=True)
    FEIL_emo_dims = ["emotion-intensity-score", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "trust"]
    
    
    Affnorms_201_df = pd.read_excel(Affective_norms_201_nouns_path)
    
    for column in Affnorms_201_df.columns:
        if "English" in column:
    
            Affnorms_201_df.drop(columns=[column], inplace=True)
    
    Affnorms_201_df["Finnish Word"] = Affnorms_201_df["Finnish Word"].str.lower().str.strip()
    Affnorms_201_df.set_index("Finnish Word", inplace=True)
    Affnorms_201_df.drop(columns=["ID", "Finnish Word Length"], inplace=True)
    
    
    Affnorms_420_df = pd.read_excel(Affective_norms_420_nouns_path, header=[0,1,2])
    Affnorms_420_df.columns = ["_".join([str(x) for x in col if pd.notna(x)]) for col in Affnorms_420_df.columns]
    Affnorms_420_df.drop(columns=["Engl. Transl._Unnamed: 1_level_1_Unnamed: 1_level_2", 'Wordlist _Unnamed: 2_level_1_Unnamed: 2_level_2'], inplace=True)
    Affnorms_420_df.rename(columns={'Finnish word_Unnamed: 0_level_1_Unnamed: 0_level_2': "Finnish Word"}, inplace=True)
    Affnorms_420_df["Finnish Word"] = Affnorms_420_df["Finnish Word"].str.lower().str.strip()
    Affnorms_420_df.set_index("Finnish Word", inplace=True)
    Affnorms_420_df = Affnorms_420_df[['Valence_All_M', 'Valence_All_SD', 'Arousal_All_M', 'Arousal_All_SD']]
    
    
    
    with open(trankit_tags_file, "rb") as f:
        trankit_tags = pickle.load(f)
    
    trankit_FA_dicts = extract_df_from_trankit(trankit_tags)
    
    FA_ids = []
    utt_lemmas = []
    
    
    for FA_dict in trankit_FA_dicts:
        
        FA_ids.append(FA_dict["utt_id"])
        lemmas = []
        
        if pd.isna(FA_dict["text"]):
            lemmas.append("")
            utt_lemmas.append(lemmas)
            continue
        
        for token in FA_dict["tokens"]:
            
            if pd.isna(token):
                lemmas.append("")
                continue
            
            if "lemma" in token.keys():
            
                lemmas.append(token["lemma"])
                
            else:
                
                lemmas.append("")
            
        utt_lemmas.append(lemmas)
    
    


    
    SELF_vectors, SELF_columns = get_lexicon_vectors(utt_lemmas, SELF_df, SELF_emo_dims)
    SELF_columns = add_lexicon_name_to_columns(SELF_columns, "SELF")
    
    with open(SELF_target_file, "wb") as f:
        pickle.dump((FA_ids, SELF_vectors, SELF_emo_dims), f)
    
    FEIL_vectors, FEIL_columns = get_lexicon_vectors(utt_lemmas, FEIL_df, FEIL_emo_dims)
    FEIL_columns = add_lexicon_name_to_columns(FEIL_columns, "FEIL")
    
    with open(FEIL_target_file, "wb") as f:
        pickle.dump((FA_ids, FEIL_vectors, FEIL_emo_dims), f)
    
    Affnorms_210_vectors, Affnorms_210_columns = get_lexicon_vectors(utt_lemmas, Affnorms_201_df, Affnorms_201_df.columns)
    Affnorms_210_columns = add_lexicon_name_to_columns(Affnorms_210_columns, "210")
    
    with open(Affnorms_210_target_file, "wb") as f:
        pickle.dump((FA_ids, Affnorms_210_vectors, Affnorms_201_df.columns), f)
        
        
    Affnorms_420_vectors, Affnorms_420_columns = get_lexicon_vectors(utt_lemmas, Affnorms_420_df, Affnorms_420_df.columns)
    Affnorms_420_columns = add_lexicon_name_to_columns(Affnorms_420_columns, "420")
    
    with open(Affnorms_420_target_file, "wb") as f:
        pickle.dump((FA_ids, Affnorms_420_vectors, Affnorms_420_df.columns), f)
    
    
    lexicon_vectors_concatenated = np.concatenate((SELF_vectors, FEIL_vectors, Affnorms_210_vectors, Affnorms_420_vectors), axis=1)
    lexicon_columns = SELF_columns+FEIL_columns+Affnorms_210_columns+Affnorms_420_columns
    
    
    with open(lexicon_vectors_target_file, "wb") as f:
        pickle.dump((FA_ids, lexicon_vectors_concatenated, lexicon_columns), f)
    
    
    
    
    