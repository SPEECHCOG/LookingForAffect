#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:10:23 2026

@author: Kalle Lahtinen

Script for extracting FinModernBERT embeddings from text data
"""

import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
import pickle



def mean_pooling(last_hidden_state, attention_mask):
    """
    last_hidden_state: (batch, seq_len, hidden_dim)
    attention_mask:   (batch, seq_len)
    """
    # Mean pooling converts token-level embeddings into a single sentence-level embedding.
    # Each token in the sentence has a hidden_dim-dimensional representation in last_hidden_state.
    # The attention mask indicates which tokens are real (1) and which are padding (0).
    #
    # By masking out padding tokens, summing over tokens, and dividing by the number of
    # real tokens, we obtain a fixed-length embedding that represents the entire sentence.
    
    # This function computes a sentence-level embedding by averaging token embeddings.
    #
    # First, the attention_mask is expanded to match the dimensionality of last_hidden_state,
    # where each token is represented by a hidden_dim-long vector
    # (1024 dimensions in the case of finnish-modernbert-large).
    #
    # Token embeddings corresponding to real tokens are multiplied by 1,
    # while PAD token embeddings are multiplied by 0 and thus excluded from the pooling.
    #
    # Next, the masked token embeddings are summed over the token dimension
    # (i.e., over all tokens within each sample), resulting in a single
    # hidden_dim-long vector per sample.
    #
    # Finally, this summed vector is divided by the number of real (non-PAD)
    # tokens in the sample, yielding the mean-pooled sentence embedding.

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked_embeddings = last_hidden_state * mask

    sum_embeddings = masked_embeddings.sum(dim=1)
    lengths = attention_mask.sum(dim=1).unsqueeze(-1)

    return sum_embeddings / lengths



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
        output_file = target_dir+"FinmodernBERT_coll_embeddings.pkl"
        

    else:
        df = pd.read_csv(data_storage_dir+"/gpt_output_fixed_only.csv", encoding="utf-8")
        texts = df["fixed_new_text"].astype(str).tolist()
        FA_ids = df["fixed_id"].tolist()
        output_file = target_dir+"FinmodernBERT_noncoll_embeddings.pkl"


    MODEL_NAME = "TurkuNLP/finnish-modernbert-large"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # important: inference mode
    
    
    
    BATCH_SIZE = 8
    all_embeddings = []
    all_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_ids = FA_ids[i:i+BATCH_SIZE]
            
            #Tokenize samples in the batch
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,  # adjust if needed
                return_tensors="pt"
            ).to(device)
    
            outputs = model(**encoded)
            
            pooled = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"]
            )
    
            all_embeddings.append(pooled.cpu().numpy())
            all_ids.extend(batch_ids)
    
    embeddings = np.vstack(all_embeddings)
    
    
    
    with open(output_file, 'wb') as f:
        pickle.dump((all_ids, embeddings), f)
    
    