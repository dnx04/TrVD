import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from src.model import BatchProgramClassifier

def parse_options():
    parser = argparse.ArgumentParser(description='TrVD training.')
    parser.add_argument('-i', '--input', default='trvd',
                        help='Dataset name (subfolder under subtrees/)', type=str)
    parser.add_argument('-m', '--model', default='rvnn-att', choices=['rvnn-att'],
                        type=str, required=False, help='sub-tree model type')
    parser.add_argument('-d', '--device', default='cuda',
                        type=str, required=False, help='Device (default: cuda)')
    parser.add_argument('-e', '--epochs', default=100,
                        help='Number of training epochs (default: 100)', type=int)
    parser.add_argument('-p', '--patience', default=5,
                        help='Early stopping patience (default: 5)', type=int)
    parser.add_argument('-b', '--batch-size', default=64,
                        help='Batch size (default: 64)', type=int)
    parser.add_argument('-w', '--workers', default=4,
                        help='DataLoader workers (default: 4)', type=int)
    args = parser.parse_args()
    return args


def collate_batch(batch):
    """Collate a batch of (code, label) tuples. Code is a list of block sequences (list of lists)."""
    data = [item[0] for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    return data, labels


class BlockDataset(torch.utils.data.Dataset):
    """Wraps a preprocessed DataFrame for DataLoader use."""

    def __init__(self, df: pd.DataFrame):
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row['code'], row['label']


if __name__ == '__main__':
    args = parse_options()
    embedding_size = 128
    w2v_path = 'subtrees/' + args.input + '/node_w2v_' + str(embedding_size)
    train_data = pd.read_pickle('subtrees/'+args.input+'/train_block.pkl')
    val_data = pd.read_pickle('subtrees/'+args.input+'/dev_block.pkl')

    # filter dataset for code is []
    train_data = train_data.drop(train_data[train_data['code'].str.len() == 0].index)
    val_data = val_data.drop(val_data[val_data['code'].str.len() == 0].index)
    print('train: \n', train_data['label'].value_counts())

    word2vec = Word2Vec.load(w2v_path).wv
    embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    ENCODE_DIM = 128
    LABELS = 86
    EPOCHS = args.epochs
    BATCH_SIZE = 32
    USE_GPU = True
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    device = torch.device("cuda")
    print(device)
    print('dataset:', args.input)
    BATCH_SIZE = args.batch_size
    print('batch size:', BATCH_SIZE)

    train_dataset = BlockDataset(train_data)
    val_dataset = BlockDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_batch,
                              pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_batch,
                            pin_memory=True, persistent_workers=args.workers > 0)

    model = BatchProgramClassifier(EMBEDDING_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                                    device, USE_GPU, embeddings)

    if USE_GPU:
        model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.8)
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    model_path = 'saved_model/' + args.input + '/' + args.model
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    best_model = 'saved_model/best_' + args.input + '.pt'

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    patience_counter = 0
    print('Start training...')
    # training procedure
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_acc = torch.tensor(0.0).to(device)
        total_loss = 0.0
        total = 0.0
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        all_train_preds = []
        all_train_labels = []
        for train_inputs, train_labels in pbar:
            train_labels = train_labels.to(device)
            model.zero_grad()
            model.batch_size = len(train_labels)
            output = model(train_inputs)
            loss = loss_function(output, train_labels)
            loss.backward()
            optimizer.step()
            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            all_train_preds += predicted.tolist()
            all_train_labels += train_labels.tolist()
            total += len(train_labels)
            total_loss += loss.item() * len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        train_acc = total_acc.item() / total
        train_prec = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_rec = recall_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        print('-' * 89)
        print('| end of epoch {:3d} / {:3d} | time: {:5.2f}s | train loss {:5.2f} | train acc {:5.2f} | train prec {:5.2f} | train rec {:5.2f} | train f1 {:5.2f} | lr {:.8f}'
              .format(epoch+1, EPOCHS, (time.time() - start_time), total_loss / total, train_acc, train_prec, train_rec, train_f1, scheduler.get_lr()[0]))

        if val_dataset is not None:
            end_time = time.time()
            all_labels = []
            all_preds = []
            total_acc = torch.tensor(0.0).to(device)
            total_loss = 0.0
            total = 0.0
            model.eval()
            for test_inputs, test_labels in tqdm(val_loader, desc="Val", leave=False):
                test_labels = test_labels.to(device)
                model.batch_size = len(test_labels)
                output = model(test_inputs)

                loss = loss_function(output, test_labels)
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == test_labels).sum()
                all_labels += test_labels.tolist()
                all_preds += predicted.tolist()

                total += len(test_labels)
                total_loss += loss.item() * len(test_inputs)

            torch.save(model.state_dict(), model_path + '/model_'+str(epoch+1)+'.pt')
            current_val_acc = total_acc.item() / total
            current_val_prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            current_val_rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            current_val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            if current_val_acc > best_acc:
                best_acc = current_val_acc
                patience_counter = 0
                print('saving best model')
                torch.save(model.state_dict(), best_model)
            else:
                patience_counter += 1
            print('| end of epoch {:3d} / {:3d} | time: {:5.2f}s | val loss {:5.2f} | val acc {:5.2f} | val prec {:5.2f} | val rec {:5.2f} | val f1 {:5.2f}'
                  .format(epoch + 1, EPOCHS, (time.time() - end_time), total_loss / total, current_val_acc, current_val_prec, current_val_rec, current_val_f1))
            print('-' * 89)
            scheduler.step()
            if patience_counter >= args.patience:
                print('-' * 89)
                print(f'Early stopping triggered after {epoch + 1} epochs (no improvement for {args.patience} epochs)')
                print('-' * 89)
                break




