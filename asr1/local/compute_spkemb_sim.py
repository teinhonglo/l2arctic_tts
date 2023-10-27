import os
import json
import soundfile
from tqdm import tqdm
import sys
import shutil
from collections import defaultdict
import argparse

import numpy as np
from numpy.linalg import norm

from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import product
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data/all_16k",
                    type=str)

parser.add_argument("--data_dir2",
                    default="data/yourtts_test",
                    type=str)

parser.add_argument("--spkids",
                    default=None,
                    type=str)

args = parser.parse_args()

data_dir = args.data_dir
data_dir2 = args.data_dir2
spkids = args.spkids

src_wavscp = os.path.join(data_dir, "wav.scp")
src_utt2spk = os.path.join(data_dir, "utt2spk")
src_utt2spk_dict = {}
src_wavs_dict = {}

tgt_wavscp = os.path.join(data_dir2, "wav.scp")
tgt_wavs_dict = {}

uttid_list = []

with open(src_wavscp, "r") as fn:
    for line in fn.readlines():
        uttid, wav_path = line.split()
        src_wavs_dict[uttid] = preprocess_wav(wav_path)
        uttid_list.append(uttid)

with open(src_utt2spk, "r") as fn:
    for line in fn.readlines():
        uttid, spkid = line.split()
        src_utt2spk_dict[uttid] = spkid

with open(tgt_wavscp, "r") as fn:
    for line in fn.readlines():
        uttid, wav_path = line.split()
        tgt_wavs_dict[uttid] = preprocess_wav(wav_path)


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

# The neural network will automatically use CUDA if it's available on your machine, otherwise it 
# will use the CPU. You can enforce a device of your choice by passing its name as argument to the 
# constructor. The model might take a few seconds to load with CUDA, but it then executes very 
# quickly.
encoder = VoiceEncoder()

if spkids is None:
    spkid_list = list(set(list(src_utt2spk_dict.values())))
else:
    spkid_list = spkids.split(",")
    
spk_sims = {spkid: [] for spkid in spkid_list}
spk_embeds = {spkid: {"embed1":[], "embed2": []} for spkid in spkid_list}

for uttid in tqdm(uttid_list):
    spkid = src_utt2spk_dict[uttid]
    src_wav_path = src_wavs_dict[uttid]
    tgt_wav_path = tgt_wavs_dict[uttid]
    
    embed1 = encoder.embed_utterance(src_wav_path)
    embed2 = encoder.embed_utterance(tgt_wav_path)
    cos_sim = cosine_similarity(embed1, embed2)
    
    spk_sims[spkid].append(cos_sim)
    
    spk_embeds[spkid]["embed1"].append(embed1)
    spk_embeds[spkid]["embed2"].append(embed2)

# utterance-wise similarity
print("utterance-wise similarity")
all_utt_sims = {"mean": [], "std": []}
for spkid in spkid_list:
    sims = np.array(spk_sims[spkid])
    mean, std = np.mean(sims), np.std(sims)
    all_utt_sims["mean"].append(mean)
    all_utt_sims["std"].append(std) 
    print("{spkid},{mean},{std}".format(spkid=spkid, mean=mean, std=std))

all_utt_sims["mean"], all_utt_sims["std"] = np.array(all_utt_sims["mean"]), np.array(all_utt_sims["std"])
print("{spkid},{mean},{std}".format(spkid="all", mean=np.mean(all_utt_sims["mean"]), std=np.mean(all_utt_sims["std"])))
print()

# speaker-wise similarity
print("speaker-wise similarity")
all_sims = {"mean": [], "std": []}
for spkid in spkid_list: 
    embed1s = np.array(spk_embeds[spkid]["embed1"])
    embed2s = np.array(spk_embeds[spkid]["embed2"])
    
    #sims = np.dot(embed1s, embed2s.T) / (np.linalg.norm(embed1s, axis=1)[:, np.newaxis] * np.linalg.norm(embed2s, axis=1))
    sims = []
    for e1, e2 in product(embed1s, embed2s):
        sims.append(cosine_similarity(e1, e2))
    
    sims = np.array(sims)
    mean, std = np.mean(sims), np.std(sims)
    all_sims["mean"].append(mean)
    all_sims["std"].append(std) 
    print("{spkid},{mean},{std}".format(spkid=spkid, mean=mean, std=std))

all_sims["mean"], all_sims["std"] = np.array(all_sims["mean"]), np.array(all_sims["std"])
print("{spkid},{mean},{std}".format(spkid="all", mean=np.mean(all_sims["mean"]), std=np.mean(all_sims["std"])))
