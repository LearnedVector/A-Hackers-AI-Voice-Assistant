"""Training script"""

import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import torch.optim as optim
from dataset import (
    WakeWordData, collate_fn,
    TripletWakeWordData, triplet_collate_fn
)
from models import LSTMWakeWord, SiameseWakeWord
from sklearn.metrics import classification_report
from tabulate import tabulate


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, model_params, notes=None):
    torch.save({
        "notes": notes,
        "model_params": model_params,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, checkpoint_path)


def test(test_loader, model, device, epoch):
    print("\n starting test for epoch %s"%epoch)
    model.eval()
    similar_distances = []
    disimilar_distances = []
    pdist = nn.PairwiseDistance(p=2)
    with torch.no_grad():
        for idx, (anchor, pos, neg) in enumerate(test_loader):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            out1, out2, out3 = model(anchor), model(pos), model(neg)

            similar_dist = pdist(out1, out2).item()
            disimilar_dist = pdist(out1, out3).item()
            similar_distances.append(similar_dist)
            disimilar_distances.append(disimilar_dist)

            print("Iter: {}/{}, sim dist: {}, dis dist: {}, dist_diff: {}".format(
                idx, len(test_loader), similar_dist, disimilar_dist,
                disimilar_dist - similar_dist), end="\r")

    avg_similar_dist = np.mean(similar_distances)
    avg_disimilar_dist = np.mean(disimilar_distances)
    sim_std = np.std(similar_distances)
    dis_std = np.std(disimilar_distances)
    return avg_similar_dist, avg_disimilar_dist, sim_std, dis_std


def train(train_loader, model, optimizer, loss_fn, device, epoch):
    print("\n starting train for epoch %s"%epoch)
    losses = []
    preds = []
    labels = []
    for idx, (anchor, pos, neg) in enumerate(train_loader):
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        out1, out2, out3 = model(anchor), model(pos), model(neg)
        loss = loss_fn(out1, out2, out3)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        print("epoch: {}, Iter: {}/{}, loss: {}".format(epoch, idx, len(train_loader), loss.item()), end="\r")
    avg_train_loss = np.mean(losses)
    print('avg train loss:', avg_train_loss)
    return avg_train_loss


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_dataset = TripletWakeWordData(data_json=args.train_data_json, sample_rate=args.sample_rate, valid=False)
    test_dataset = TripletWakeWordData(data_json=args.test_data_json, sample_rate=args.sample_rate, valid=True)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        collate_fn=lambda x: triplet_collate_fn(x, valid=False),
                                        **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                        batch_size=args.eval_batch_size,
                                        shuffle=True,
                                        collate_fn=lambda x: triplet_collate_fn(x, valid=True),
                                        **kwargs)

    model_params = {
        "embedding_size": 5,
        "filter_size": 16,
        "feature_size": 40,
        "dropout": 0.1
    }
    model = SiameseWakeWord(**model_params)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_epoch = 0
    best_loss = float("inf")
    best_table = None
    for epoch in range(args.epochs):
        print("\nstarting training with learning rate", optimizer.param_groups[0]['lr'])
        loss = train(train_loader, model, optimizer, loss_fn, device, epoch)
        sim_dist, disim_dist, std_sim, std_dism = test(test_loader, model, device, epoch)

        table = [["Train loss", loss],
                ["Test avg sim distance", sim_dist],
                ["Test avg disim distance", disim_dist],
                ["Test dist diff", disim_dist - sim_dist],
                ["Test std sim ", std_sim],
                ["Test std disim", std_dism]]

        print("\n\n** metrics table **")
        print(tabulate(table))

        # # saves checkpoint if metrics are better than last
        if args.save_checkpoint_path and loss < best_loss:

            checkpoint_path = os.path.join(args.save_checkpoint_path, args.model_name + ".pt")
            print("\nfound best checkpoint. saving model as", checkpoint_path)
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, model_params,
                notes="train_loss: {}, test_dist_diff: {}, epoch: {}".format(loss, disim_dist - sim_dist, epoch),
            )
            best_loss = loss
            best_epoch = epoch
            best_table = table

        scheduler.step(loss)

    print("Done Training...")
    print("Best Model Saved to", checkpoint_path)
    print("Best Epoch", best_epoch)
    print("\n\n** metrics table **")
    print(tabulate(best_table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wake Word Training Script")
    parser.add_argument('--sample_rate', type=int, default=8000, help='sample_rate for data')
    parser.add_argument('--epochs', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model_name', type=str, default="wakeword", required=False, help='name of model to save')
    parser.add_argument('--save_checkpoint_path', type=str, default=None, help='Path to save the best checkpoint')
    parser.add_argument('--train_data_json', type=str, default=None, required=True, help='path to train data json file')
    parser.add_argument('--test_data_json', type=str, default=None, required=True, help='path to test data json file')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--hidden_size', type=int, default=128, help='lstm hidden size')

    args = parser.parse_args()

    main(args)
