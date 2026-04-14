# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrVD (abstract syntax Tree decomposition based Vulnerability Detector) is a deep learning vulnerability detection system for C/C++ source code. It decomposes ASTs into ordered sub-trees, encodes each with a tree RNN, and aggregates with a Transformer encoder to classify 86 vulnerability types (85 CWE types + benign).

## Commands

```bash
# Install dependencies (pre-configured for CUDA-enabled torch)
uv sync

# Verify GPU access
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Split dataset into train/val/test (stratified, default seed=220703)
uv run python scripts/split_dataset.py -i dataset/dataset.pkl -o ./dataset/trvd

# Preprocess: AST parsing, Word2Vec training, block sequence generation
uv run python -m src.pipeline --input trvd --output subtrees/trvd

# Train model
uv run python scripts/train.py

# Evaluate on test set
uv run python scripts/evaluation.py

# Normalize raw source code (if starting from raw data)
uv run python -m src.normalization -i ./dataset/trvd -o ./dataset/trvd
```

## Architecture

```
Raw Code → Normalization → AST (tree-sitter) → AST Decomposition (sub-trees)
                                                          ↓
                                              Word2Vec (gensim, skip-gram)
                                                          ↓
                                              Block Sequences (token indices)
                                                          ↓
                                              BatchTreeEncoder (GRU + attention)
                                                          ↓
                                              Transformer Encoder (4 heads, 2 layers)
                                                          ↓
                                              Max Pool → FC → 86-class softmax
```

### Data Flow

1. **Normalization** (`src/normalization.py`): Strips comments, removes string literals, renames identifiers to `VAR_i` / `FUN_i`
2. **AST Decomposition** (`src/pipeline.py` + `src/prepare_data.py`): Parses code with tree-sitter, recursively decomposes into sub-trees (max depth=8, max size=40 nodes)
3. **Word2Vec Training** (`src/pipeline.py`): Trains on root-to-leaf paths + statement sequences from training ASTs
4. **Block Sequence Generation** (`src/pipeline.py`): Converts each sub-tree to a list of vocabulary indices
5. **Model** (`src/model.py`):
   - `BatchTreeEncoder`: Bottom-up GRU with attention over child nodes → fixed-dim vector per sub-tree
   - `BatchProgramClassifier`: Pads/truncates sub-tree sequences to uniform length, passes through Transformer encoder, max-pools across sub-trees, classifies with FC layer
6. **Evaluation** (`scripts/evaluation.py`): Runs inference on test set, reports accuracy/precision/recall/F1

### Key Source Files

| File | Purpose |
|------|---------|
| `src/model.py` | `BatchTreeEncoder` (AttRvNN) + `BatchProgramClassifier` (Transformer encoder) |
| `src/prepare_data.py` | `get_blocks` (AST decomposition), `get_root_paths` (path collection for Word2Vec), `needsSplitting` (depth/size check) |
| `src/pipeline.py` | `Pipeline` class — orchestrates parse → Word2Vec → block sequence generation |
| `src/tree.py` | `ASTNode` (sub-tree wrapper), `SingleNode` (leaf-only wrapper) |
| `src/normalization.py` | Code normalization (identifier renaming, comment/string removal) |
| `src/clean_gadget.py` | Helper for code cleaning during normalization |
| `scripts/train.py` | Training loop: Adamax lr=0.001, StepLR gamma=0.8 every 10 epochs, early stopping |
| `scripts/evaluation.py` | Test evaluation with weighted precision/recall/F1 |
| `scripts/split_dataset.py` | Stratified train/val/test split with configurable seed |

### Dataset Format

Pickle DataFrames with columns:
- `code`: normalized C/C++ source code (string)
- `label`: integer 0–85 (0=benign, 1–85=CWE types)

Expected layout:
```
dataset/
  dataset.pkl        # raw combined dataset
  trvd/              # split output (use scripts/split_dataset.py to create)
    train.pkl        # 80% stratified split
    val.pkl          # 10%
    test.pkl         # 10%
```

Output from pipeline:
```
subtrees/{input}/
  node_w2v_128        # Word2Vec model
  train_block.pkl     # block sequences (indices)
  dev_block.pkl
  test_block.pkl
```

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Word2Vec embedding | 128 dim |
| Sub-tree encode dim | 128 |
| Transformer heads | 4 |
| Transformer layers | 2 |
| Batch size | 32 (train), 100 (eval) |
| Max sub-tree depth | 8 |
| Max sub-tree size | 40 |
| Classes | 86 |

## Important Notes

- **Python 3.12+ required**
- **torch 2.0+** — `Variable` was removed in torch 2.0; all tensor operations use raw tensors directly
- **gensim 4.x** — use `vector_size=` not `size=` in Word2Vec constructor
- **tree-sitter** has no type stubs — import as `import tree_sitter  # type: ignore[import]` and use `# type: ignore[return-value]` on `get_token()` return statements
- **CWE labels**: See `cwe_labels.csv` for label→CWE mapping and sample counts
- **Saved models**: `saved_model/best_{input}.pt` is the best validation model; per-epoch snapshots in `saved_model/{input}/{model}/model_{epoch}.pt`
