"""Utility script to create the training and test json files for speechrecognition.


"""
import os
import argparse
import json
import random

def main(args):
    data = []
    directory = args.file_folder_directory
    filetxtname = args.file_folder_directory.rpartition('/')[2]
    percent = args.percent
    
    with open(directory + "/" + filetxtname + "-metadata.txt") as f: 
        for line in f: 
            file_name = line.partition('|')[0]
            text = line.split('|')[1] 
            data.append({
            "key": directory + "/" + file_name,
            "text": text
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
    Utility script to create the training and test json files for speechrecognition. """
    )
    parser.add_argument('--file_folder_directory', type=str, default=None, required=True,
                        help='directory of clips given by mimic-recording-studio')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    
    
    
    args = parser.parse_args()

    main(args)

