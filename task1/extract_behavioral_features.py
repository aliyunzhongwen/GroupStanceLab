# -*- coding: utf-8 -*-
"""
Extract behavioral features from 877 user CSV files.
Generates feature matrix shape=(878, 8) with index 0 as zero-placeholder.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==================== Path Config ====================
NODE_INFO_CSV = "/root/CORDGT/CorDGT/processed/node_id_info_clear_new2_TF_filtered.csv"
CSV_DIR = "/root/CORDGT/CorDGT/processed/csv_ok_lable/"
OUTPUT_DIR = "/root/CORDGT/CorDGT/lab3/task1/features/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 1. Build node_id mapping ====================
print("=" * 60)
print("Step 1: Read node info CSV, build twitter_name -> node_id mapping")
print("=" * 60)

node_df = pd.read_csv(NODE_INFO_CSV)
# Row index + 1 = node_id (1-877)
name_to_id = {}
for idx, row in node_df.iterrows():
    name = str(row["twitter_name"]).strip()
    node_id = idx + 1
    name_to_id[name] = node_id

print(f"  Mapping count: {len(name_to_id)}")
print(f"  node_id range: {min(name_to_id.values())} ~ {max(name_to_id.values())}")

# ==================== 2. Extract behavioral features ====================
print("\n" + "=" * 60)
print("Step 2: Iterate CSV files, extract behavioral features")
print("=" * 60)

feature_names = [
    "tweet_count",
    "original_ratio",
    "retweet_ratio",
    "reply_ratio",
    "mention_frequency",
    "active_time_span",
    "interaction_diversity",
    "avg_tweet_length",
]

# Initialize feature matrix: shape=(878, 8), index 0 is zero-placeholder
num_nodes = 877
raw_features = np.zeros((num_nodes + 1, len(feature_names)), dtype=np.float64)

csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith(".csv")])
print(f"  CSV file count: {len(csv_files)}")

processed = 0
skipped = 0

for csv_file in csv_files:
    # Parse twitter_name from filename: {twitter_name}+{twitter_id}.csv
    base_name = csv_file[:-4]
    plus_pos = base_name.rfind("+")
    if plus_pos == -1:
        print(f"  [WARNING] Cannot parse filename: {csv_file}, skip")
        skipped += 1
        continue

    twitter_name = base_name[:plus_pos]

    # Lookup node_id
    if twitter_name not in name_to_id:
        print(f"  [WARNING] Name not in mapping: {twitter_name} (file: {csv_file}), skip")
        skipped += 1
        continue

    node_id = name_to_id[twitter_name]

    # Read CSV
    csv_path = os.path.join(CSV_DIR, csv_file)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding="latin-1")
        except Exception as e:
            print(f"  [WARNING] Read failed: {csv_file}, error: {e}, skip")
            skipped += 1
            continue
    except Exception as e:
        print(f"  [WARNING] Read failed: {csv_file}, error: {e}, skip")
        skipped += 1
        continue

    if len(df) == 0:
        processed += 1
        continue

    # --- Extract features ---
    total = len(df)

    # tweet_count
    tweet_count = float(total)

    # original_ratio: RT_ID is NaN AND R_ID is NaN
    is_original = df["RT_ID"].isna() & df["R_ID"].isna()
    original_ratio = is_original.sum() / total

    # retweet_ratio: RT_ID is not NaN
    retweet_ratio = df["RT_ID"].notna().sum() / total

    # reply_ratio: R_ID is not NaN
    reply_ratio = df["R_ID"].notna().sum() / total

    # mention_frequency: M_ID is not NaN
    mention_frequency = df["M_ID"].notna().sum() / total

    # active_time_span: timelable max - min
    time_labels = df["timelable"].dropna()
    if len(time_labels) > 0:
        active_time_span = float(time_labels.max() - time_labels.min())
    else:
        active_time_span = 0.0

    # interaction_diversity: unique users in RT_ID + M_ID + R_ID
    interact_users = set()
    for val in df["RT_ID"].dropna():
        interact_users.add(str(val).strip())
    for val in df["R_ID"].dropna():
        interact_users.add(str(val).strip())
    for val in df["M_ID"].dropna():
        mentions = str(val).split(";")
        for m in mentions:
            m = m.strip()
            if m:
                interact_users.add(m)
    interaction_diversity = float(len(interact_users))

    # avg_tweet_length: mean character count of tweets
    tweet_lengths = df["tweets"].dropna().astype(str).str.len()
    avg_tweet_length = tweet_lengths.mean() if len(tweet_lengths) > 0 else 0.0

    # Write to feature matrix
    raw_features[node_id] = [
        tweet_count,
        original_ratio,
        retweet_ratio,
        reply_ratio,
        mention_frequency,
        active_time_span,
        interaction_diversity,
        avg_tweet_length,
    ]

    processed += 1
    if processed % 100 == 0:
        print(f"  Processed {processed}/{len(csv_files)} files...")

print(f"\n  Done: success={processed}, skipped={skipped}")

# ==================== 3. Feature statistics summary ====================
print("\n" + "=" * 60)
print("Step 3: Raw feature statistics summary (node_id 1-877)")
print("=" * 60)

raw_data = raw_features[1:]  # exclude row 0 placeholder
for i, fname in enumerate(feature_names):
    col = raw_data[:, i]
    print(f"  {fname:25s}: min={col.min():10.4f}, max={col.max():10.4f}, "
          f"mean={col.mean():10.4f}, std={col.std():10.4f}")

# ==================== 4. Feature standardization ====================
print("\n" + "=" * 60)
print("Step 4: StandardScaler standardization")
print("=" * 60)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(raw_data)

# Assemble standardized matrix: shape=(878, 8), row 0 is zero vector
standard_features = np.zeros((num_nodes + 1, len(feature_names)), dtype=np.float64)
standard_features[1:] = scaled_data

print(f"  Standardized matrix shape: {standard_features.shape}")
for i, fname in enumerate(feature_names):
    col = standard_features[1:, i]
    print(f"  {fname:25s}: mean={col.mean():8.4f}, std={col.std():8.4f}")

# ==================== 5. Save results ====================
print("\n" + "=" * 60)
print("Step 5: Save feature files")
print("=" * 60)

raw_path = os.path.join(OUTPUT_DIR, "behavioral_features_raw.npy")
std_path = os.path.join(OUTPUT_DIR, "behavioral_features.npy")
names_path = os.path.join(OUTPUT_DIR, "behavioral_feature_names.json")

np.save(raw_path, raw_features)
print(f"  Raw features saved: {raw_path}, shape={raw_features.shape}")

np.save(std_path, standard_features)
print(f"  Standardized features saved: {std_path}, shape={standard_features.shape}")

with open(names_path, "w", encoding="utf-8") as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=2)
print(f"  Feature names saved: {names_path}")

print("\n[DONE] Behavioral feature extraction complete!")
