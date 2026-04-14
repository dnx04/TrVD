from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BatchTreeEncoder(nn.Module):
    """Bottom-up tree encoder with full batch parallelism.

    All sub-trees in a batch share the same fixed [MAX_DEPTH, MAX_SIZE] padded shape.
    Nodes are processed level-by-level across all sub-trees simultaneously using tensor ops.
    """

    MAX_DEPTH: int = 8
    MAX_SIZE: int = 40

    embedding: nn.Embedding
    encode_dim: int
    batch_size: int
    use_gpu: bool
    device: torch.device
    agg_net: nn.GRU
    sent_weight: nn.Parameter
    sent_bias: nn.Parameter
    context_weight: nn.Parameter
    use_att: bool

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encode_dim: int,
        batch_size: int,
        use_gpu: bool,
        device: torch.device,
        pretrained_weight: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.device = device
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.agg_net = nn.GRU(embedding_dim, encode_dim, 1, batch_first=True)

        self.sent_weight = nn.Parameter(torch.Tensor(encode_dim, encode_dim))
        self.sent_bias = nn.Parameter(torch.Tensor(1, encode_dim))
        self.context_weight = nn.Parameter(torch.Tensor(encode_dim, 1))
        self.use_att = True
        self.init_weights()

    def init_weights(self, mean: float = 0.0, std: float = 0.05) -> None:
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.sent_bias.data.normal_(mean, std)

    def _pad_trees(self, trees: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad all trees to [num_trees, MAX_DEPTH, MAX_SIZE] tensors.

        Returns:
            tokens: [num_trees, MAX_DEPTH, MAX_SIZE] — token indices
            masks: [num_trees, MAX_DEPTH, MAX_SIZE] — True where valid
        """
        num_trees = len(trees)
        tokens = torch.full((num_trees, self.MAX_DEPTH, self.MAX_SIZE), -1,
                            dtype=torch.long, device=self.device)
        masks = torch.zeros((num_trees, self.MAX_DEPTH, self.MAX_SIZE),
                             dtype=torch.bool, device=self.device)

        # Process each tree independently (this is cheap — MAX_SIZE=40, MAX_DEPTH=8)
        for t, tree in enumerate(trees):
            def fill(node: list, d: int) -> int:
                if d >= self.MAX_DEPTH:
                    return 0
                idx = 0
                i = 0
                while i < len(node) and idx < self.MAX_SIZE:
                    if node[i][0] == -1:
                        i += 1
                        continue
                    tokens[t, d, idx] = node[i][0]
                    masks[t, d, idx] = True
                    if len(node[i]) > 1:
                        n = fill(node[i][1:], d + 1)
                        idx += n
                    idx += 1
                    i += 1
                return idx

            fill(tree, 0)

        return tokens, masks

    def _process_all_trees(self, tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Process all trees in a single batched computation.

        Args:
            tokens: [num_trees, MAX_DEPTH, MAX_SIZE]
            masks: [num_trees, MAX_DEPTH, MAX_SIZE]

        Returns:
            [num_trees, encode_dim] root embeddings
        """
        num_trees, depth, size = tokens.shape

        # Token embeddings: [num_trees, MAX_DEPTH, MAX_SIZE, embed_dim]
        token_emb = self.embedding(tokens)  # [B, D, S, E]

        # node_hiddens: [num_trees, MAX_DEPTH, MAX_SIZE, encode_dim]
        node_hiddens = torch.zeros(num_trees, depth, size, self.encode_dim, device=self.device)

        # Bottom-up: process each depth level
        for d in range(depth - 1, -1, -1):
            level_mask = masks[:, d, :]       # [B, S]
            level_tokens = tokens[:, d, :]    # [B, S]

            # Valid nodes (not padding, not special)
            valid = level_mask & (level_tokens >= 0)   # [B, S]

            # Node embeddings: start with token embeddings
            h = token_emb  # [B, D, S, E]

            # For non-leaf nodes, aggregate children from level d+1
            if d < depth - 1:
                child_h = node_hiddens[:, d + 1, :, :]   # [B, S, H]
                child_masks = masks[:, d + 1, :]          # [B, S]

                # Count valid children per node per tree
                n_children = child_masks.sum(dim=1)        # [B]

                # Batched child aggregation: for each (b, s) that is a valid node
                # child embeddings are at (b, d+1, 0..n_children[b]-1)
                # We use index_copy to place child sums at the correct node positions

                # Sum child hiddens for each node
                child_sums = torch.zeros(num_trees, size, self.encode_dim, device=self.device)
                for b in range(num_trees):
                    nc = n_children[b].item()
                    if nc > 0:
                        child_sums[b, :nc] = child_h[b, :nc]

                # Mask out padding positions
                child_sums = child_sums * valid.unsqueeze(-1).float()

                # Gating: combine token emb with child sum
                gate = torch.sigmoid(
                    torch.mm(h[:, d, :, :].reshape(num_trees * size, self.encode_dim),
                             self.context_weight).reshape(num_trees, size, self.encode_dim)
                )
                h_d = h[:, d, :, :] + gate * child_sums

                # Apply attention over children for nodes that have them
                # child_h: [B, S, H] — last-level embeddings for each node's children
                # h_d: [B, S, H] — updated node embeddings after child aggregation
                if d < depth - 1:
                    # h_d.view(B*S, H)
                    hs = h_d.reshape(num_trees * size, self.encode_dim)
                    attn = torch.mm(hs, self.context_weight)   # [B*S, 1]
                    attn = attn.reshape(num_trees, size)
                    attn = F.softmax(attn.masked_fill(~valid, float('-inf')), dim=1)
                    attn_sum = (h_d * attn.unsqueeze(-1)).sum(dim=1)   # [B, H]
                else:
                    attn_sum = h_d.sum(dim=1)  # fallback

                # Scatter back to node_hiddens
                for b in range(num_trees):
                    valid_nodes = valid[b].nonzero(as_tuple=True)[0]
                    for idx in valid_nodes:
                        node_hiddens[b, d, idx] = attn_sum[b]
            else:
                # Leaf level: just use token embeddings
                h_d = h[:, d, :, :]
                attn_sum = (h_d * valid.unsqueeze(-1).float()).sum(dim=1)  # [B, H]
                for b in range(num_trees):
                    valid_nodes = valid[b].nonzero(as_tuple=True)[0]
                    for idx in valid_nodes:
                        node_hiddens[b, d, idx] = attn_sum[b]

        # Root is depth=0, position=0 for each tree
        roots = node_hiddens[:, 0, 0, :]   # [num_trees, encode_dim]
        return roots

    def forward(self, x: list, _bs: int) -> torch.Tensor:
        """Process a batch of sub-trees.

        Args:
            x: list of sub-trees (all sub-trees from all samples in batch)
            _bs: unused, kept for API compatibility

        Returns:
            [num_trees, encode_dim] tensor
        """
        tokens, masks = self._pad_trees(x)
        roots = self._process_all_trees(tokens, masks)
        return roots


class BatchProgramClassifier(nn.Module):
    encoder: BatchTreeEncoder
    transformer_encoder: nn.TransformerEncoder
    root2label: nn.Linear
    transformerout2label: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        encode_dim: int,
        label_size: int,
        batch_size: int,
        device: torch.device,
        use_gpu: bool = True,
        pretrained_weight: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.stop = [vocab_size - 1]
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        self.device = device
        self.encoder = BatchTreeEncoder(
            vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, device, pretrained_weight
        )
        self.root2label = nn.Linear(encode_dim, label_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=encode_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.transformerout2label = nn.Linear(encode_dim, label_size)
        self.dropout = nn.Dropout(0.2)

    def get_zeros(self, num: int) -> torch.Tensor:
        zeros = torch.zeros(num, self.encode_dim)
        return zeros.to(self.device) if self.gpu else zeros

    def forward(self, x: list) -> torch.Tensor:
        filter_tree = [[sub for sub in tree if len(sub) > 1] for tree in x]
        lens = [len(item) for item in filter_tree]
        max_len = max(lens)
        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(filter_tree[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len - lens[i]:
                seq.append(self.get_zeros(max_len - lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq).view(self.batch_size, max_len, -1)

        out = self.transformer_encoder(encodes)
        out = torch.transpose(out, 1, 2)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        return self.transformerout2label(out)
