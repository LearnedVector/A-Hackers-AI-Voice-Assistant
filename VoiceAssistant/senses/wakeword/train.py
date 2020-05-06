"""Training script"""

import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from dataset import WakeWordData
from model import LSTMWakeWord


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc


def test(test_loader, model, device):
    accs = []
    for idx, (mfcc, label) in enumerate(test_loader):
        mfcc, label = mfcc.to(device), label.to(device)
        output = model(mfcc)
        pred = F.sigmoid(output)
        acc = binary_accuracy(pred, label)
        accs.append(acc)
        print("Iter: {}/{}, accuracy: {}".format(idx, len(test_loader), acc))
    print('Average test Accuracy:', sum(accs)/len(accs))


def train(train_loader, model, optimizer, loss_fn, device):
    losses = []
    for idx, (mfcc, label) in enumerate(train_loader):
        mfcc, label = mfcc.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(mfcc)
        pred = F.sigmoid(output)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print("Iter: {}/{}, accuracy: {}".format(idx, len(train_loader), loss.item()))
    print('Average train loss:', sum(losses)/len(losses))


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_dataset = WakeWordData(data_json=args.train_data_json, sample_rate=args.sample_rate)
    test_dataset = WakeWordData(data_json=args.test_data_json, sample_rate=args.sample_rate)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        **kwargs)

    model = LSTMWakeWord(feature_size=40, hidden_size=128, num_layers=1, dropout=0.1, bidirectional=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    for i in range(args.epochs):
        train(train_loader, model, optimizer, loss_fn, device)
        test(test_loader, model, device)
        # checkpoint model here


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wake Word Training Script")
    parser.add_argument('--sample_rate', type=int, default=8000, help='sample_rate for data')
    parser.add_argument('--epochs', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--train_data_json', type=str, default=None, required=True,
                        help='path to train data json file')
    parser.add_argument('--test_data_json', type=str, default=None, required=True,
                        help='path to test data json file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num_workers', type=str, default=1, default=False,
                        help='number of data loading workers')
    args = parser.parse_args()

    main(args)
