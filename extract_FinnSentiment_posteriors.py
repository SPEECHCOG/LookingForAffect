#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:10:23 2026

@author: Kalle Lahtinen

Script for extracting FinnSentiment posteriors from text data.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
import torch.nn.functional as F
import pickle





if __name__ == '__main__':

    # ------------------------
    # Parse command-line or default arguments
    # ------------------------
    if len(sys.argv) < 2:
        # default path if no arguments provided
        data_storage_dir = "./data/"
        target_dir = "./data/"
        colloquial = True
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_storage_dir", nargs=1, type=str)
        parser.add_argument("--target_dir", nargs=1, type=str)
        parser.add_argument("--colloquial", nargs=1, type=bool)
        args = parser.parse_args()
        data_storage_dir = args.dataset_storage_dir[0]
        target_dir = args.target_dir[0]
        colloquial = args.colloquial[0]
        
        
        
    if colloquial:
        df = pd.read_csv(data_storage_dir+"/text_LP_filtered_annotated.csv", encoding="utf-8")
        texts = df["transcript"].astype(str).tolist()
        FA_ids = df["Unnamed: 0"].tolist()
        output_file = target_dir+"Finnsentiment_coll_posteriors.pkl"

    else:
        df = pd.read_csv(data_storage_dir+"/gpt_output_fixed_only.csv", encoding="utf-8")
        texts = df["fixed_new_text"].astype(str).tolist()
        FA_ids = df["fixed_id"].tolist()
        output_file = target_dir+"Finnsentiment_noncoll_posteriors.pkl"

    

    
    MODEL_NAME = "fergusq/finbert-finnsentiment"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # important: inference mode
    
    
    
    BATCH_SIZE = 8
    all_sentiments = []
    all_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_ids = FA_ids[i:i+BATCH_SIZE]
            
            #Tokenize samples in the batch
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True, # adjust if needed
                max_length=512,
                return_tensors="pt"
            ).to(device)
    
            logits = model(**encoded).logits
            posteriors = F.softmax(logits, dim=1).detach().cpu().numpy()
            
            all_sentiments.append(posteriors)
            all_ids.extend(batch_ids)
    
    sentiments = np.vstack(all_sentiments)
    
    with open(output_file, 'wb') as f:
        pickle.dump((all_ids, sentiments), f)