import os
import sys
sys.path.append('/share/nas167/teinhonglo/github_repo/charsiu/src')
from Charsiu import charsiu_forced_aligner

import subprocess
import json
import re
import soundfile
from tqdm import tqdm
import numpy as np
import shutil
import ast

import argparse


# ArgumentParser
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data-mdd/all",
                    type=str)

parser.add_argument("--output_dir",
                    default="data-mdd/all/align/org",
                    type=str)

parser.add_argument("--model_path",
                    default="charsiu/en_w2v2_fc_10ms",
                    type=str)

parser.add_argument("--wavscp_fn",
                    default="wav.scp",
                    type=str)

args = parser.parse_args()

# args
data_dir = args.data_dir
output_dir = args.output_dir
model_path = args.model_path
wavscp_fn = args.wavscp_fn

print("output dir: {output_dir}".format(output_dir=output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 
else:
    print("{output_dir} had already existed".format(output_dir=output_dir))
    #exit(0)

# Data Preparation
src_wavscp = os.path.join(data_dir, wavscp_fn)
src_text = os.path.join(data_dir, "text")

uttid_list = []
src_wavscp_dict = {}
src_text_dict = {}

with open(src_wavscp, "r") as fn:
    for line in fn.readlines():
        uttid, wav_path = line.split()
        src_wavscp_dict[uttid] = wav_path
        uttid_list.append(uttid)
        
with open(src_text, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        uttid = info[0]
        text = " ".join(info[1:])
        src_text_dict[uttid] = text

# initialize model
charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms', sil_threshold=4, cost_thres=0.0)

easy_tried_sils = [(2, 0.0), (1, 0.0)]
hard_tried_sils = [(6, 0.0), (10, 0.0), (15, 0.0), (15, 0.8), (15, 0.9), (20, 0.9), (200, 0.99), (2000, 0.999)]

for uttid in tqdm(uttid_list):
    wav_path = src_wavscp_dict[uttid]
    text = src_text_dict[uttid]
    
    textgrid_fn = os.path.join(output_dir, uttid + ".TextGrid")

    if os.path.exists(textgrid_fn):
        continue
    
    owrds = text.split()
    num_owrds = len(owrds)
    
    # sanic check
    # perform forced alignment
    aphns, awrds = charsiu.align(audio=wav_path, text=text, ref_phones=phones)
    num_awrds = sum(1 for _, _, word in awrds if word != '[SIL]')
    num_aphns = sum(1 for _, _, phone in aphns if phone != '[SIL]')

    if num_owrds < num_awrds:
        tried_sils = hard_tried_sils
    else:       
        tried_sils = easy_tried_sils

    tried_idx = -1
    
    while num_owrds != num_awrds:
        try:
            tried_idx += 1
            charsiu.sil_threshold, charsiu.cost_thres = tried_sils[tried_idx]
        except:
            tried_idx -= 1
            break
        
        aphns, awrds = charsiu.align(audio=wav_path, text=text)
        num_awrds = sum(1 for _, _, word in awrds if word != '[SIL]')
        num_aphns = sum(1 for _, _, phone in aphns if phone != '[SIL]')
    
    if num_wrds != num_wrds:
        print()
        print("aligner.py")
        print("owrds", owrds)
        print("awrds_text", [p for _, _, p in awrds if p != "[SIL]"])
        print("awrds", aphns)
        print("tried_idx", tried_idx, tried_sils[tried_idx])
        input()
        
    # perform forced alignment and save the output as a textgrid file
    charsiu.serve(audio=wav_path,
                  text=text,
                  save_to=textgrid_fn)
    
    if tried_idx != -1:
        print("Tried", tried_sils[tried_idx], ". Resetting now.")
        charsiu.sil_threshold, charsiu.cost_thres = 4, 0.0
