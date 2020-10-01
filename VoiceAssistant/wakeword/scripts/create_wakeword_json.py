"""Utility script to create training json file for wakeword.

    There should be two directories. one that has all of the 0 labels
    and one with all the 1 labels
"""
import os
import argparse
import json
import random


def main(args):
    zeros = os.listdir(args.zero_label_dir)
    ones = os.listdir(args.one_label_dir)

    data = []
    for z in zeros:
        data.append({
            "key": os.path.join(args.zero_label_dir, z),
            "label": 0
        })
    for o in ones:
        data.append({
            "key": os.path.join(args.one_label_dir, o),
            "label": 1
        })
    random.shuffle(data)

    with open(args.save_json_path, 'w') as f:
        for d in data:
            line = json.dumps(d)
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to create training json file for wakeword.

    There should be two directories. one that has all of the 0 labels
    and one with all the 1 labels
    """
    )
    parser.add_argument('--zero_label_dir', type=str, default=None, required=True,
                        help='directory of clips with zero labels')
    parser.add_argument('--one_label_dir', type=str, default=None, required=True,
                        help='directory of clips with one labels')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to save json file')

    args = parser.parse_args()

    main(args)
