from neuralnet.utils import TextProcess
import ctcdecode
import torch

textprocess = TextProcess()

labels = [
    "'",  # 0
    " ",  # 1
    "a",  # 2
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # 27
    "_",  # 28, blank
]

def DecodeGreedy(output, blank_label=28, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2).squeeze(1)
	decode = []
	for i, index in enumerate(arg_maxes):
		if index != blank_label:
			if collapse_repeated and i != 0 and index == arg_maxes[i -1]:
				continue
			decode.append(index.item())
	return textprocess.int_to_text_sequence(decode)

class CTCBeamDecoder:

    def __init__(self, beam_size=100, blank_id=labels.index('_'), kenlm_path=None):
        print("loading beam search with lm...")
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels, alpha=0.522729216841, beta=0.96506699808,
            beam_width=beam_size, blank_id=labels.index('_'),
            model_path=kenlm_path)
        print("finished loading beam search")

    def __call__(self, output):
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])
