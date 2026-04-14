"""
Normalize C/C++ source code:
- Strips comments (// and /* */)
- Removes string and character literals
- Renames user-defined identifiers to VAR_i / FUN_i
- Adds normalized code back to the DataFrame
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd
from tqdm import tqdm
from src.clean_gadget import clean_gadget


def normalization(source):
    nor_code = []
    for fun in tqdm(source["code"], desc="Normalizing"):
        lines = fun.split("\n")
        code = ""
        for line in lines:
            line = line.strip()
            line = re.sub(r"//.*", "", line)
            code += line + " "
        code = re.sub(r"/\*.*?\*/", "", code)
        code = clean_gadget([code])
        nor_code.append(code[0])
    return nor_code


def parse_options():
    parser = argparse.ArgumentParser(description="Normalize dataset source code")
    parser.add_argument(
        "--input",
        "-i",
        default="dataset/splits",
        help="Input directory containing train.pkl, val.pkl, test.pkl (default: dataset/splits)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: same as --input)",
    )
    parser.add_argument(
        "--name",
        "-n",
        default=None,
        help="Output folder name under dataset/. Overrides --output if set. (e.g. 'normalized' → dataset/normalized/)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_options()

    input_dir = args.input
    output_dir = args.output if args.output else (os.path.join("dataset", args.name) if args.name else input_dir)

    print(f"Normalizing dataset from: {input_dir}/")
    for split in ["train", "val", "test"]:
        src = os.path.join(input_dir, f"{split}.pkl")
        dst = os.path.join(output_dir, f"{split}.pkl")
        if not os.path.exists(src):
            print(f"  SKIP {split}.pkl (not found)")
            continue
        df = pd.read_pickle(src)
        print(f"\n[{split}] Normalizing {len(df)} samples...")
        df["code"] = normalization(df)
        os.makedirs(output_dir, exist_ok=True)
        df.to_pickle(dst)
        print(f"  Saved to {dst}")

    print(f"\nNormalization complete. Output: {output_dir}/")


if __name__ == "__main__":
    main()
