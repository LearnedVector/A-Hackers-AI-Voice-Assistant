"""Freezes and optimize the model. Use after training."""
import argparse
import torch
from models import LSTMWakeWord, SiameseWakeWord

def get_model(checkpoint, model):
    if model == 'lstm':
        return LSTMWakeWord(**checkpoint['model_params'], device='cpu')
    if model == 'siamese':
        return SiameseWakeWord(**checkpoint['model_params'])

def trace(model):
    model.eval()
    x = torch.rand(80, 1, 40)
    traced = torch.jit.trace(model, (x))
    return traced

def main(args):
    print("loading model from", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    model = get_model(checkpoint, args.model)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("tracing model...")
    traced_model = trace(model)
    print("saving to", args.save_path)
    traced_model.save(args.save_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the wakeword engine")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True,
                        help='Checkpoint of model to optimize')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='path to save optmized model')
    parser.add_argument('--model', type=str, default='lstm', required=True,
                        help='lstm or siamed. default: lstm')

    args = parser.parse_args()
    main(args)
