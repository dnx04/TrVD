from __future__ import annotations

import argparse
import copy
import os

import pandas as pd
from tqdm import tqdm
from tree_sitter import Language, Parser
import tree_sitter_cpp as tspp
from src.prepare_data import get_root_paths

def parse_options():
    parser = argparse.ArgumentParser(description='TrVD preprocess~.')
    parser.add_argument('-i', '--input', default='splits',
                        help='Dataset folder name under dataset/ (default: splits)', type=str)
    parser.add_argument('-o', '--output', default='subtrees',
                        help='Output folder for artifacts (default: subtrees)', type=str)
    args = parser.parse_args()
    return args


def parse_ast(source: str) -> object:
    cpp_lang = Language(tspp.language())
    parser = Parser(cpp_lang)  # type: ignore[operator]
    tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
    return tree

args = parse_options()

class Pipeline:
    def __init__(self):
        self.train = None
        self.train_keep = None
        self.train_block = None
        self.dev = None
        self.dev_keep = None
        self.dev_block = None
        self.test = None
        self.test_keep = None
        self.test_block = None
        self.size = None
        self.w2v_path = None

    # parse source code
    def parse_source(self, dataset):
        data_dir = os.path.join('dataset', dataset)
        train = pd.read_pickle(os.path.join(data_dir, 'train.pkl'))
        test = pd.read_pickle(os.path.join(data_dir, 'test.pkl'))
        dev = pd.read_pickle(os.path.join(data_dir, 'val.pkl'))

        # parsing source into ast
        print("Parsing train...")
        train['code'] = [parse_ast(c) for _, c in tqdm(train['code'].items(), total=len(train))]
        self.train = train
        self.train_keep = copy.deepcopy(train)
        print("Parsing dev...")
        dev['code'] = [parse_ast(c) for _, c in tqdm(dev['code'].items(), total=len(dev))]
        self.dev = dev
        self.dev_keep = copy.deepcopy(dev)
        print("Parsing test...")
        test['code'] = [parse_ast(c) for _, c in tqdm(test['code'].items(), total=len(test))]
        self.test = test
        self.test_keep = copy.deepcopy(test)


    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, size: int) -> None:
        self.size = size
        trees = self.train
        assert trees is not None, "parse_source() must be called before dictionary_and_embedding()"
        self.w2v_path = os.path.join(args.output, args.input, 'node_w2v_' + str(size))
        if not os.path.exists(os.path.join(args.output, args.input)):
            os.makedirs(os.path.join(args.output, args.input), exist_ok=True)
        from src.prepare_data import get_sequences
        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            # collect all root-leaf paths
            paths = []
            get_root_paths(ast, paths, [])
            # add root to leaf path as corpus
            paths.append(sequence)
            return paths
        # train word2vec embedding if not exists
        if not os.path.exists(self.w2v_path):
            print("Collecting root-to-leaf paths for Word2Vec...")
            corpus = [trans_to_sequences(ast) for _, ast in tqdm(trees['code'].items(), total=len(trees))]
            paths = []
            for all_paths in corpus:
                for path in all_paths:
                    path = [token.decode('utf-8') if type(token) is bytes else token for token in path]
                    paths.append(path)
            corpus = paths
            # training word2vec model
            from gensim.models.word2vec import Word2Vec
            print('corpus size: ', len(corpus))
            w2v = Word2Vec(corpus, vector_size=size, workers=96, sg=1, min_count=3)
            print('word2vec : ', w2v)
            w2v.save(self.w2v_path)

    # generate block sequences with index representations
    def generate_block_seqs(self, data, name: str) -> pd.DataFrame:
        blocks_path: str
        if name == 'train':
            blocks_path = os.path.join(args.output, args.input, 'train_block.pkl')
        elif name == 'test':
            blocks_path = os.path.join(args.output, args.input, 'test_block.pkl')
        else:
            blocks_path = os.path.join(args.output, args.input, 'dev_block.pkl')

        from src.prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec
        word2vec = Word2Vec.load(self.w2v_path).wv
        vocab = word2vec.vocab
        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token = node.token
            if type(token) is bytes:
                token = token.decode('utf-8')
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def tree_to_token(node):
            token = node.token
            if type(token) is bytes:
                token = token.decode('utf-8')
            result = [token]
            children = node.children
            for child in children:
                result.append(tree_to_token(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        trees = data
        print(f"Generating block sequences ({name})...")
        trees['code'] = [trans2seq(c) for _, c in tqdm(trees['code'].items(), total=len(trees))]
        trees.to_pickle(blocks_path)
        return trees


    # run for processing raw to train
    def run(self, dataset):
        print('parse source code...')
        self.parse_source(dataset)
        print('train word2vec model...')
        self.dictionary_and_embedding(size=128)
        print('generate block sequences...')
        self.train_block = self.generate_block_seqs(self.train_keep, 'train')
        self.dev_block = self.generate_block_seqs(self.dev_keep, 'dev')
        self.test_block = self.generate_block_seqs(self.test_keep, 'test')


if __name__ == '__main__':
    ppl = Pipeline()
    print('Now precessing dataset: ', args.input)
    ppl.run(args.input)


