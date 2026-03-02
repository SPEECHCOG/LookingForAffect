#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 2026

@author: Kalle Lahtinen

Script to extract ExHuBERT embeddings from wav files.
Each wav file should be named according to its FA_id.
"""

import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
import pickle
import soundfile as sf
import os


if __name__ == '__main__':

    # ------------------------
    # Parse command-line or default arguments
    # ------------------------
    if len(sys.argv) < 2:
        wav_dir = "/Path/To/FinnAffect/root/wavs/"  # folder containing .wav files
        data_storage_dir = "./data/"
        output_file = data_storage_dir+"/ExHuBERT_embeddings_test.pkl"
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--wav_dir", nargs=1, type=str)
        parser.add_argument("--data_storage_dir", nargs=1, type=str)
        args = parser.parse_args()
        wav_dir = args.wav_dir[0]
        data_storage_dir = args.data_storage_dir[0]
        output_file = data_storage_dir+"/ExHuBERT_embeddings.pkl"

    # ------------------------
    # Get list of wav files and FA_ids
    # ------------------------
    
    df = pd.read_csv(data_storage_dir+"/text_LP_filtered_annotated.csv", encoding="utf-8")
    FA_ids = df["Unnamed: 0"].tolist()
    wav_files = []

    for FA_id in FA_ids:
        wav_files.append(wav_dir+str(FA_id)+".wav")


    # ------------------------
    # Load feature extractor and model
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960"
    )
    model = AutoModel.from_pretrained(
        "amiriparian/ExHuBERT",
        trust_remote_code=True
    )
    
    print("Using device: ")
    print(device)
    model.to(device)
    model.eval()

    # ------------------------
    # Extraction loop
    # ------------------------
    BATCH_SIZE = 4  # can adjust depending on GPU memory
    all_embeddings = []
    all_ids = []
    
    print("Embedding extraction starting")

    with torch.no_grad():
        for i in tqdm(range(0, len(wav_files), BATCH_SIZE)):
            batch_files = wav_files[i:i+BATCH_SIZE]
            batch_ids = FA_ids[i:i+BATCH_SIZE]

            batch_speech = []
            for f in batch_files:
                speech, sr = sf.read(os.path.join(wav_dir, f))
                if sr != 16000:
                    print("Wrong SR, quit")
                    break
                batch_speech.append(speech.astype(np.float32))

            # Extract features
            inputs = feature_extractor(
                batch_speech,
                sampling_rate=16000,
                padding = 'max_length',
                max_length = 20*sr,
                return_tensors="pt",
                return_attention_mask=True
            ).to(device)

            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            pooled = hidden_states.mean(dim=1)

            all_embeddings.append(pooled.cpu().numpy())
            all_ids.extend(batch_ids)
            

    embeddings = np.vstack(all_embeddings)

    # ------------------------
    # Save embeddings
    # ------------------------
    with open(output_file, 'wb') as f:
        pickle.dump((all_ids, embeddings), f)

    print(f"Saved ExHuBERT embeddings for {len(all_ids)} samples to {output_file}")