from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def matrix_mul(input: torch.Tensor, weight: torch.Tensor, bias: torch.nn.parameter.Parameter | None = None) -> torch.Tensor:
    feature_list = []
    for feature in input:
        feat = torch.mm(feature, weight)
        if bias is not None:
            feat = feat + bias.expand(feat.size(0), bias.size(1))
        feat = torch.tanh(feat).unsqueeze(0)
        feature_list.append(feat)
    return torch.cat(feature_list, 0).squeeze(-1)


def element_wise_mul(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feat = feature_1 * feature_2
        feature_list.append(feat.unsqueeze(0))
    output = torch.cat(feature_list, 0)
    return torch.sum(output, 0).unsqueeze(0)


class BatchTreeEncoder(nn.Module):
    embedding: nn.Embedding
    encode_dim: int
    batch_size: int
    use_gpu: bool
    node_list: list[torch.Tensor]
    batch_node: torch.Tensor
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
        self.node_list = []
        self.device = device
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.agg_net = nn.GRU(embedding_dim, encode_dim, 1)

        self.sent_weight = nn.Parameter(torch.Tensor(encode_dim, encode_dim))
        self.sent_bias = nn.Parameter(torch.Tensor(1, encode_dim))
        self.context_weight = nn.Parameter(torch.Tensor(encode_dim, 1))
        self.use_att = True
        self.init_weights()

    def init_weights(self, mean: float = 0.0, std: float = 0.05) -> None:
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.sent_bias.data.normal_(mean, std)

    def create_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device) if self.use_gpu else tensor

    def traverse_mul(self, node: list, batch_index: list[int]) -> torch.Tensor | None:
        size = len(node)
        if not size:
            return None

        batch_current = self.create_tensor(torch.zeros(size, self.encode_dim, dtype=torch.float32))

        index: list[int] = []
        children_index: list[list[int]] = []
        current_node: list[int] = []
        children: list[list] = []

        for i in range(size):
            if node[i][0] != -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] != -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        idx_tensor = torch.tensor(index, dtype=torch.long, device=self.device)
        emb_tensor = self.embedding(torch.tensor(current_node, dtype=torch.long, device=self.device))
        batch_current = batch_current.index_copy(0, idx_tensor, emb_tensor)

        childs_hidden_sum = self.create_tensor(torch.zeros(size, self.encode_dim, dtype=torch.float32))

        hidden_per_child: list[torch.Tensor] = []
        for c in range(len(children)):
            zeros = self.create_tensor(torch.zeros(size, self.encode_dim, dtype=torch.float32))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                child_idx = torch.tensor(children_index[c], dtype=torch.long, device=self.device)
                cur_child_hidden = zeros.index_copy(0, child_idx, tree)
                childs_hidden_sum += cur_child_hidden
                hidden_per_child.append(cur_child_hidden)

        if self.use_att and len(hidden_per_child) != 0:
            child_hiddens = torch.stack(hidden_per_child)
            childs_weighted = matrix_mul(child_hiddens, self.sent_weight, self.sent_bias)
            childs_weighted = matrix_mul(childs_weighted, self.context_weight).permute(1, 0)
            childs_weighted = F.softmax(childs_weighted, dim=-1)
            childs_hidden_sum = element_wise_mul(child_hiddens, childs_weighted.permute(1, 0)).squeeze(0)

        batch_current = batch_current.unsqueeze(0)
        childs_hidden_sum = childs_hidden_sum.unsqueeze(0)
        _, hn = self.agg_net(batch_current, childs_hidden_sum)
        hn = hn.squeeze(0)

        batch_index = [i for i in batch_index if i != -1]
        b_in = torch.tensor(batch_index, dtype=torch.long, device=self.device)
        nd_tmp = self.batch_node.index_copy(0, b_in, hn)
        self.node_list.append(nd_tmp)

        return hn

    def forward(self, x: list, bs: int) -> torch.Tensor:
        self.batch_size = bs
        self.batch_node = self.create_tensor(torch.zeros(self.batch_size, self.encode_dim, dtype=torch.float32))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        node_list_stacked = torch.stack(self.node_list)
        return torch.max(node_list_stacked, 0)[0]


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