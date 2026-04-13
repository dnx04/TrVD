# TrVD: Deep Semantic Extraction via AST Decomposition for Vulnerability Detection

TrVD exposes indicative semantics deeply embedded in source code fragments for accurate and efficient C/C++ vulnerability detection, using a divide-and-conquer strategy. It decomposes ASTs into ordered sub-trees of restricted sizes/depths, encodes each sub-tree with a tree-structured neural network, and summarizes them with a Transformer encoder to capture long-range semantic relationships.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12.6 (e.g., H100, A100)
- NVIDIA Driver >= 535

## Installation

```bash
# Clone and set up the project
uv sync
```

This installs all dependencies. The `pyproject.toml` is pre-configured to fetch the **CUDA-enabled PyTorch build** from the PyTorch wheel index (`cu126` — CUDA 12.6), so `torch.cuda.is_available()` will return `True` on any machine with a compatible NVIDIA GPU and driver.

### Verify GPU Access

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Dataset

The project uses the [SARD dataset](https://samate.nist.gov/SRD/index.php) — 264,822 C/C++ functions with 86 vulnerability classes (85 vulnerable types + 1 benign).

Place your dataset pickle files at:
```
dataset/
  train.pkl    # training set
  val.pkl       # validation set
  test.pkl      # test set
```

Each pickle contains a DataFrame with columns:
- `code`: normalized C/C++ source code (string)
- `label`: integer 0–85 (0 = benign, 1–85 = CWE types)

A merged dataset is available at `dataset/dataset.pkl`.

### CWE Label Reference

See [cwe_labels.csv](cwe_labels.csv) for the mapping of label numbers to CWE types and sample counts.

## Pipeline

### Step 1: Normalize Code

Preprocess raw source code by removing comments, string/char literals, and normalizing identifiers.

```bash
uv run python normalization.py
```

### Step 2: AST Decomposition

Parse normalized code into ASTs, decompose into sub-trees of max depth 8 and max size 40, train Word2Vec embeddings, and generate block sequences.

```bash
uv run python pipeline.py
```

This produces:
- `subtrees/trvd/node_w2v_128` — trained Word2Vec model
- `subtrees/trvd/train_block.pkl` — training block sequences
- `subtrees/trvd/dev_block.pkl` — validation block sequences
- `subtrees/trvd/test_block.pkl` — test block sequences

### Step 3: Train

Train the TrVD model (Tree RNN + Transformer encoder) on the block sequences.

```bash
uv run python train.py
```

Checkpoints saved to:
- `saved_model/trvd/rvnn-att/model_<epoch>.pt` — per-epoch snapshots
- `saved_model/best_trvd.pt` — best validation model

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | `trvd` | Dataset name (subfolder under `subtrees/`) |
| `-m, --model` | `rvnn-att` | Model type |
| `-d, --device` | `cuda` | Device (`cuda`, `cuda:1`, `cuda:2`, or `cpu`) |

### Step 4: Evaluate

Evaluate the trained model on the test set.

```bash
uv run python evaluation.py
```

Outputs accuracy, precision, recall, and F1-score across all 86 classes.

## Architecture

```
Raw Code
  │
  ▼
Code Normalization      (normalization.py)
  │                     Strips comments, literals, normalizes identifiers
  ▼
AST Parsing             (pipeline.py / tree-sitter)
  │                     Parses C/C++ into abstract syntax trees
  ▼
AST Decomposition       (prepare_data.py)
  │                     Splits AST into sub-trees (depth≤8, size≤40)
  ▼
Word2Vec Embedding      (gensim / pipeline.py)
  │                     Trains token embeddings from root-to-leaf paths
  ▼
Tree RNN Encoder        (model.py / BatchTreeEncoder)
  │                     Encodes each sub-tree into a fixed-dim vector (GRU)
  ▼
Transformer Encoder     (model.py / nn.TransformerEncoder)
  │                     Captures long-range relationships among sub-trees
  ▼
Max Pooling + FC        (model.py / BatchProgramClassifier)
  │
  ▼
86-class Classification  (vulnerability type or benign)
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Word2Vec embedding size | 128 |
| Sub-tree encode dimension | 128 |
| LSTM hidden dimension | 100 |
| Transformer attention heads | 4 |
| Transformer layers | 2 |
| Batch size | 32 (train), 100 (eval) |
| Optimizer | Adamax (lr=0.001, step_lr gamma=0.8 every 10 epochs) |
| Max sub-tree depth | 8 |
| Max sub-tree size | 40 |
| Dropout | 0.2 |

## Performance

The model classifies C/C++ functions into 86 vulnerability classes including:
- **CWE-119**: Buffer Errors (27,571 samples — largest class)
- **CWE-763**: Use of Out-of-range Pointer Offset
- **CWE-77**: Command/Argument Injection
- **CWE-22**: Path Traversal
- And 82 more vulnerability types

## References

- Paper: [TrVD: Deep Semantic Extraction via AST Decomposition for Vulnerability Detection](docs/TrVD.pdf)
- Dataset: [NIST SARD](https://samate.nist.gov/SRD/index.php)
- Tree-sitter: [tree-sitter-c](languages/tree-sitter-c), [tree-sitter-cpp](languages/tree-sitter-cpp)
