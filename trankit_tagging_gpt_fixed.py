#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:23:06 2026

@author: lahtine9
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trankit batch processing script for large corpora.

Features:
- Processes text samples in batches of 100k
- Stores results in per-batch pickle files
- Skips already-processed samples
- Atomic writes to avoid file corruption
- Compatible with HPC/Slurm
"""

# ------------------------
# Imports
# ------------------------
import sys
import pandas as pd
import argparse
from trankit import Pipeline
import pickle
from pathlib import Path

skip_trankit = False

# ------------------------
# Initialize Trankit pipeline
# ------------------------

local_model_path = "/Volumes/T9/trankit_cache/models/v1.0.0"

if not skip_trankit:
    #trankit_pipe = Pipeline('finnish')  # load the Finnish model once
    trankit_pipe = Pipeline(lang="finnish", cache_dir=local_model_path)
# ------------------------
# Trankit wrapper function
# ------------------------
def trankit_pipeline(transcript):
    """
    Process a single transcript with Trankit and return full output.
    Can modify this to return only lemmas, tokens, etc.
    """
    output = trankit_pipe(transcript, is_sent=True)
    return output

# ------------------------
# Main
# ------------------------
if __name__ == '__main__':

    # ------------------------
    # Parse command-line or default arguments
    # ------------------------
    if len(sys.argv) < 2:
        # default path if no arguments provided
        data_storage_dir = "/Path/To/FinnAffect/root/"
        target_dir = "./data/"
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_storage_dir", nargs=1, type=str)
        args = parser.parse_args()
        data_storage_dir = args.dataset_storage_dir[0]

    # ------------------------
    # Load CSV of transcripts
    # ------------------------
    

    
    gpt_output_df = pd.read_csv(target_dir+"gpt_output_fixed_only.csv")
    
    
    text_filtered_list = gpt_output_df["fixed_new_text"]
    text_sample_ids = gpt_output_df["fixed_id"]
    
    output_file = target_dir+"trankit_gpt_fixed.pkl"
    results = []
    
    for i, sample in enumerate(text_filtered_list):
        
        print(str(i)+"//"+str(len(text_filtered_list)))
    
        sample_id = text_sample_ids[i]
        # Process sample
        
        if pd.isna(sample):
            results.append((sample_id, sample))
            continue
        
        if not skip_trankit:
            try:
                result = trankit_pipeline(sample)
                results.append((sample_id, result))
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
      
    if not skip_trankit:    
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
    
    # Save batch atomically
        #puhti-login12
    print("✅ All samples processed.")
    

    
    
    
    
    