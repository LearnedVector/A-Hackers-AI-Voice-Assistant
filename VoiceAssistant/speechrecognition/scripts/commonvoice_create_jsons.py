"""Utility script to create the training and test json files for speechrecognition.


"""
import os
import argparse
import json
import random
import csv

def main(args):
    data = []
    directory = args.file_path.rpartition('/')[0]
    percent = args.percent
    with open(args.file_path, newline='') as csvfile: 
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:  
            file_name = row['path']
            text = row['sentence']
            data.append({
            "key": directory + "/clips/" + file_name,
            "text": text
            })
    
    random.shuffle(data)
    
    f = open(args.save_json_path +"/"+ "train.json", "w")
    
    with open(args.save_json_path +"/"+ 'train.json','w') as f:
        d = len(data)
        i=0
        while(i<d-d/percent):
            r=data[i-1]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
    
    f = open(args.save_json_path +"/"+ "test.json", "w")

    with open(args.save_json_path +"/"+ 'test.json','w') as f:
        d = len(data)
        i=int(d-d/percent)
        while(i<d):
            r=data[i-1]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to create the training and test json files for speechrecognition. """
    )
    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    
    
    
    args = parser.parse_args()

    main(args)

