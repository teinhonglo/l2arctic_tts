import os
import sys

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


args = parser.parse_args()

# args
data_dir = args.data_dir


# Data Preparation
src_text = os.path.join(data_dir, "text")
src_phn = os.path.join(data_dir, "transcript_phn_text")
src_phn_wb = os.path.join(data_dir, "transcript_phn_text_wb")

uttid_list = []
src_wavscp_dict = {}
src_text_dict = {}
src_phn_dict = {}
src_phn_wb_dict = {}

        
with open(src_text, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        uttid = info[0]
        text = " ".join(info[1:])
        src_text_dict[uttid] = text
        uttid_list.append(uttid)

with open(src_phn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        uttid = info[0]
        text = " ".join(info[1:])
        src_phn_dict[uttid] = text

with open(src_phn_wb, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        uttid = info[0]
        text = " ".join(info[1:])
        src_phn_wb_dict[uttid] = text


for uttid in uttid_list:
    text = src_text_dict[uttid]
    text_phn = " ".join(re.sub("sil", "", src_phn_dict[uttid]).split())
    text_phn_wb = ast.literal_eval(src_phn_wb_dict[uttid])

    text_phn_wb_nowb = []
    
    for tpw in text_phn_wb:
        for tp in tpw:
            text_phn_wb_nowb.append(re.sub("[0-9]", "", tp).lower())
    text_phn_wb_nowb = " ".join(text_phn_wb_nowb)

    if len(text_phn.split()) != len(text_phn_wb_nowb.split()):
        print(uttid)
        print(text_phn)
        print(text_phn_wb_nowb)
        print(text_phn_wb)
