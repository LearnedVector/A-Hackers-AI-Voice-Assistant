"""Freezes and optimize the model. Use after training."""
import argparse
import torch
from model import LSTMWakeWord

def trace(model):
    model.eval()
    x = torch.rand(80, 1, 40)
    traced = torch.jit.trace(model, (x))
    return traced

def main(args):
    print("loading model from", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    model = LSTMWakeWord(**checkpoint['model_params'], device='cpu')
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

    args = parser.parse_args()
    main(args)
