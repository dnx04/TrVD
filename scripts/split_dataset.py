"""
Split a dataset pickle into stratified train/val/test splits.
Usage: uv run python scripts/split_dataset.py -i dataset/dataset.pkl -o ./dataset/trvd
Default seed: 220703
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_options():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input pickle file (e.g. dataset/dataset.pkl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output folder path (e.g. ./dataset/trvd)",
    )
    parser.add_argument(
        "--train-ratio",
        "-t",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        "-v",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=220703,
        help="Random seed (default: 220703)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_options()

    df = pd.read_pickle(args.input)
    print(f"Loaded {len(df)} samples from {args.input}")

    label_counts = df["label"].value_counts()
    singleton_labels = set(label_counts[label_counts == 1].index.tolist())
    if singleton_labels:
        print(f"Singleton classes (forced to train): {sorted(singleton_labels)}")

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = 1.0 - train_ratio - val_ratio

    if test_ratio <= 0:
        raise ValueError("train-ratio + val-ratio must be < 1.0")

    # Separate singleton classes (cannot be stratified-split across val/test)
    if singleton_labels:
        singleton_mask = df["label"].isin(singleton_labels)
        train_df = df[singleton_mask].copy()
        temp_df = df[~singleton_mask].copy()
    else:
        train_df = pd.DataFrame()
        temp_df = df

    # Stratified split on the rest
    if len(temp_df) > 0:
        strat_train, strat_temp = train_test_split(
            temp_df,
            train_size=train_ratio,
            stratify=temp_df["label"],
            random_state=args.seed,
        )
        train_df = pd.concat([train_df, strat_train], ignore_index=True)
        # Split temp into val/test — use non-stratified since rare classes
        # may have too few samples to split evenly (e.g. labels with only 3-6 samples)
        val_ratio_adjusted = val_ratio / (test_ratio + val_ratio)
        val_df, test_df = train_test_split(
            strat_temp,
            train_size=val_ratio_adjusted,
            random_state=args.seed,
        )
    else:
        raise ValueError("No samples available after removing singleton classes")

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_pickle(os.path.join(output_dir, "train.pkl"))
    val_df.to_pickle(os.path.join(output_dir, "val.pkl"))
    test_df.to_pickle(os.path.join(output_dir, "test.pkl"))

    print(f"\nSplits saved to {output_dir}/:")
    print(f"  train.pkl: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  val.pkl:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  test.pkl:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Seed: {args.seed}")


if __name__ == "__main__":
    main()
