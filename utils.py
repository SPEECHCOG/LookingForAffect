#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 09:00:42 2026

@author: lahtine9
"""

import pandas as pd
import numpy as np

def process_expanded_samples(tokens_list):
    
    processed_tokens_list = []
    
    for token in tokens_list:
        
        if "expanded" in token.keys():
            
            for expanded_token in token["expanded"]:
                 expanded_token["expanded from"] = token["text"]
                 
                 processed_tokens_list.append(expanded_token)
                
        else:
            
            processed_tokens_list.append(token)
        
    return processed_tokens_list

def extract_df_from_trankit(trankit_tags):
    trankit_data = []
    for FA_sample_id, FA_sample in trankit_tags:
        
        try: 
            if pd.isna(FA_sample):
                trankit_data.append({"utt_id": int(FA_sample_id),
                                     "text": np.nan,
                                     "tokens": np.nan})
                
            else:
                
                if int(FA_sample_id) == 201:
                    print("")
                
                tokens_list = process_expanded_samples(FA_sample["tokens"])
                
                trankit_data.append({"utt_id": int(FA_sample_id),
                                     "text": FA_sample["text"],
                                     "tokens": tokens_list})
        
        except TypeError as e:
            print(e)
            return "type error"
            
    return trankit_data


            