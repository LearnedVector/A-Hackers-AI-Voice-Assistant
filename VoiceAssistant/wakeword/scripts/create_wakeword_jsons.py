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
    percent = args.percent
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

    f = open(args.save_json_path +"/"+ "train.json", "w")
    
    with open(args.save_json_path +"/"+ 'train.json','w') as f:
        d = len(data)
        i=0
        while(i<int(d-d/percent)):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
    
    f = open(args.save_json_path +"/"+ "test.json", "w")

    with open(args.save_json_path +"/"+ 'test.json','w') as f:
        d = len(data)
        i=int(d-d/percent)
        while(i<d):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
    

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
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    args = parser.parse_args()

    main(args)
