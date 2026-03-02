#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 2026

@author: Kalle Lahtinen

Script to extract openSMILE eGeMAPS features from wav files.
Each wav file should be named according to its FA_id.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
import pickle
import soundfile as sf
import os

import opensmile


if __name__ == '__main__':

    # ------------------------
    # Parse command-line or default arguments
    # ------------------------
    if len(sys.argv) < 2:
        wav_dir = "/Path/To/FinnAffect/root/wavs/"
        data_storage_dir = "./data/"
        output_file = data_storage_dir + "/eGeMAPS_features.pkl"
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--wav_dir", nargs=1, type=str)
        parser.add_argument("--data_storage_dir", nargs=1, type=str)
        args = parser.parse_args()
        wav_dir = args.wav_dir[0]
        data_storage_dir = args.data_storage_dir[0]
        output_file = data_storage_dir + "/eGeMAPS_features.pkl"

    # ------------------------
    # Get list of wav files and FA_ids
    # ------------------------

    df = pd.read_csv(data_storage_dir + "/text_LP_filtered_annotated.csv", encoding="utf-8")
    FA_ids = df["Unnamed: 0"].tolist()
    wav_files = []

    for FA_id in FA_ids:
        wav_files.append(wav_dir + str(FA_id) + ".wav")

    # ------------------------
    # Load openSMILE feature extractor
    # ------------------------

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # ------------------------
    # Extraction loop
    # ------------------------

    all_embeddings = []
    all_ids = []

    print("eGeMAPS feature extraction starting")

    for wav_path, fa_id in tqdm(zip(wav_files, FA_ids), total=len(FA_ids)):
        if not os.path.exists(wav_path):
            print(f"Missing file: {wav_path}")
            continue

        try:
            # openSMILE reads the file internally
            feats_df = smile.process_file(wav_path)
            feats = feats_df.values.squeeze()

            all_embeddings.append(feats)
            all_ids.append(fa_id)

        except Exception as e:
            print(f"Failed processing {wav_path}: {e}")

    embeddings = np.vstack(all_embeddings)

    # ------------------------
    # Save features
    # ------------------------

    with open(output_file, 'wb') as f:
        pickle.dump((all_ids, embeddings), f)

    print(f"Saved eGeMAPS features for {len(all_ids)} samples to {output_file}")