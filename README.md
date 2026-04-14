# TrVD: Deep Semantic Extraction via AST Decomposition for Vulnerability Detection

TrVD exposes indicative semantics deeply embedded in source code fragments for accurate and efficient C/C++ vulnerability detection, using a divide-and-conquer strategy. It decomposes ASTs into ordered sub-trees of restricted sizes/depths, encodes each sub-tree with a tree-structured neural network, and summarizes them with a Transformer encoder to capture long-range semantic relationships.

## Requirements

- Python 3.12+
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

A merged dataset is available at `dataset/dataset.pkl`. Use `scripts/split_dataset.py` to split it into train/val/test.

### Split the Dataset

```bash
uv run python scripts/split_dataset.py -i dataset/dataset.pkl -o ./dataset/trvd
```

This produces stratified splits at 80/10/10 (seed: 220703):
```
dataset/trvd/
  train.pkl    # ~211,857 samples
  val.pkl      # ~26,482 samples
  test.pkl     # ~26,483 samples
```

### Normalize the Dataset

If starting from raw (non-normalized) source code:

```bash
uv run python -m src.normalization -i ./dataset/trvd -o ./dataset/trvd
```

Each pickle contains a DataFrame with columns:
- `code`: normalized C/C++ source code (string)
- `label`: integer 0–85 (0 = benign, 1–85 = CWE types)

### CWE Label Reference

See [cwe_labels.csv](dataset/cwe_labels.csv) for the mapping of label numbers to CWE types and sample counts.

## Pipeline

Preprocesses normalized code: parses into ASTs, decomposes into sub-trees, trains Word2Vec embeddings, and generates block sequences.

```bash
uv run python -m src.pipeline --input trvd --output subtrees
```

This produces:
```
subtrees/trvd/
  node_w2v_128      # trained Word2Vec model
  train_block.pkl   # training block sequences
  dev_block.pkl     # validation block sequences
  test_block.pkl    # test block sequences
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | `trvd` | Dataset folder under `dataset/` |
| `-o, --output` | `subtrees` | Output folder for artifacts |

## Train

```bash
uv run python -m scripts.train -i trvd
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | — | Dataset name (subfolder under `subtrees/`) |
| `-m, --model` | `rvnn-att` | Model type |
| `-d, --device` | `cuda` | Device (`cuda` or `cpu`) |
| `-e, --epochs` | `100` | Number of training epochs |
| `-p, --patience` | `5` | Early stopping patience |

Checkpoints saved to:
- `saved_model/trvd/rvnn-att/model_<epoch>.pt` — per-epoch snapshots
- `saved_model/best_trvd.pt` — best validation model

## Evaluate

```bash
uv run python -m scripts.evaluation -i trvd
```

Outputs accuracy, precision, recall, and F1-score across all 86 classes.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | — | Dataset name (subfolder under `subtrees/`) |

## Architecture

```
Raw Code
  │
  ▼
Code Normalization          (src/normalization.py)
  │                         Strips comments, literals, normalizes identifiers
  ▼
AST Parsing                 (src/pipeline.py + tree-sitter-cpp)
  │                         Parses C/C++ into abstract syntax trees
  ▼
AST Decomposition           (src/prepare_data.py)
  │                         Splits AST into sub-trees (depth≤8, size≤40)
  ▼
Word2Vec Embedding          (src/pipeline.py + gensim)
  │                         Trains token embeddings from root-to-leaf paths
  ▼
Tree RNN Encoder            (src/model.py / BatchTreeEncoder)
  │                         Encodes each sub-tree into a fixed-dim vector (GRU + attention)
  ▼
Transformer Encoder         (src/model.py / nn.TransformerEncoder)
  │                         Captures long-range relationships among sub-trees
  ▼
Max Pooling + FC            (src/model.py / BatchProgramClassifier)
  │
  ▼
86-class Classification     (vulnerability type or benign)
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Word2Vec embedding size | 128 |
| Sub-tree encode dimension | 128 |
| GRU hidden dimension | 100 |
| Transformer attention heads | 4 |
| Transformer layers | 2 |
| Batch size | 32 (train), 100 (eval) |
| Optimizer | Adamax (lr=0.001, step_lr gamma=0.8 every 10 epochs) |
| Max sub-tree depth | 8 |
| Max sub-tree size | 40 |
| Dropout | 0.2 |

## Project Structure

```
.
├── src/                    # Core library
│   ├── __init__.py
│   ├── clean_gadget.py    # Identifier normalization (VAR_i, FUN_i)
│   ├── model.py           # BatchTreeEncoder + BatchProgramClassifier
│   ├── normalization.py    # Code normalization script
│   ├── pipeline.py        # AST parsing, Word2Vec, block sequence generation
│   ├── prepare_data.py    # AST decomposition (get_blocks, get_root_paths)
│   └── tree.py            # ASTNode, SingleNode wrappers
├── scripts/
│   ├── train.py           # Training entry point
│   ├── evaluation.py      # Evaluation entry point
│   └── split_dataset.py   # Stratified dataset splitter
├── dataset/
│   ├── dataset.pkl        # Raw combined dataset
│   ├── cwe_labels.csv     # Label → CWE mapping
│   └── trvd/              # Split output (from split_dataset.py)
├── build_languages/        # (removed — tree-sitter-cpp installed via pip)
└── docs/
    └── TrVD.md            # Paper reference
```

## Performance

The model classifies C/C++ functions into 86 vulnerability classes including:
- **CWE-119**: Buffer Errors (27,571 samples — largest class)
- **CWE-763**: Use of Out-of-range Pointer Offset
- **CWE-77**: Command/Argument Injection
- **CWE-22**: Path Traversal
- And 82 more vulnerability types

## References

- Paper: [docs/TrVD.md](docs/TrVD.md)
- Dataset: [NIST SARD](https://samate.nist.gov/SRD/index.php)
