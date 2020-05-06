"""Training script"""

import argparse
import torch
from dataset import WakeWordData
from model import LSTMWakeWord


def test():
    pass


def train():
    pass


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_dataset = WakeWordData(data_json=args.train_data_json, sample_rate=args.sample_rate)
    test_dataset = WakeWordData(data_json=args.test_data_json, sample_rate=args.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wake Word Training Script")
    parser.add_argument('--sample_rate', type=int, default=8000,
                        help='sample_rate for data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size of batch')
    parser.add_argument('--train_data_json', type=str, default=None, required=True,
                        help='path to train data json file')
    parser.add_argument('--test_data_json', type=str, default=None, required=True,
                        help='path to test data json file')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    main(args)