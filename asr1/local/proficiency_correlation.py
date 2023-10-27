import os
import json
import soundfile
from tqdm import tqdm
import numpy as np
import sys
import shutil
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data/all_16k",
                    type=str)

parser.add_argument("--log_fn",
                    default="data/all_16k/dtw/all_16k/result.log",
                    type=str)

args = parser.parse_args()

data_dir = args.data_dir
log_fn = args.log_fn
result_dir = os.path.dirname(log_fn)

spk2tofel_fn = os.path.join(data_dir, "spk2tofel")

def get_stats(numeric_list, prefix=""):
    # number, mean, standard deviation (std), median, mean absolute deviation
    stats_np = np.array(numeric_list)
    number = len(stats_np)
    
    if number == 0:
        summ = 0.
        mean = 0.
        std = 0.
        median = 0.
        mad = 0.
        maximum = 0.
        minimum = 0.
    else:
        summ = np.sum(stats_np)
        mean = np.mean(stats_np)
        std = np.std(stats_np)
        median = np.median(stats_np)
        mad = np.sum(np.absolute(stats_np - mean)) / number
        maximum = np.max(stats_np)
        minimum = np.min(stats_np)
    
    stats_dict = {  
#                    prefix + "number": number,
                    prefix + "mean": mean, 
                    prefix + "std": std, 
                    prefix + "median": median, 
                    prefix + "mad": mad, 
                    prefix + "summ": summ,
                    prefix + "max": maximum,
                    prefix + "min": minimum
                 }
    return stats_dict

spk2tofel_dict = {}
log_dict = {}
feat_types_list = ["waveform", "spectrogram", "wav2vec2", "whisper"]

with open(spk2tofel_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        spkid = info[0]
        spk2tofel_dict[spkid] = int(info[1])

with open(log_fn, "r") as fn:
    for line in fn.readlines():
        if "Utterance ID:" in line:
            uttid = line.split()[-1]
            spkid = uttid.split("_")[0]
            if spkid not in log_dict:
                log_dict[spkid] = {ft: [] for ft in feat_types_list}
            continue
        
        info, dtw = line.split(":")
        feat_type = info.split()[-1]
        log_dict[spkid][feat_type].append(float(dtw))


spk2stats_dict = {}
valid_spkid_list = list(set(list(log_dict.keys())).intersection(set(list(spk2tofel_dict.keys()))))

# 統計量
for spkid in valid_spkid_list:
    tofel = spk2tofel_dict[spkid]
    
    spk2stats_dict[spkid] = {}
    
    for feat_type in feat_types_list:
        stats_dict = get_stats(log_dict[spkid][feat_type])
        spk2stats_dict[spkid][feat_type] = stats_dict


# 計算相關係數 (SPC/PCC)
from scipy import stats
import matplotlib.pyplot as plt

def plot(list1, list2, xlabel, ylabel, title, fpath):
    sorted_pairs = sorted(zip(list1, list2))
    sorted_list1, sorted_list2 = map(list, zip(*sorted_pairs))
    
    plt.scatter(x=sorted_list1, y=sorted_list2)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fpath)
    plt.close()

stats_list = list(stats_dict.keys())
stats_fn = open(os.path.join(result_dir, "stats.log"), "w")
stats_fn.write("Valid SpkeID " + str(len(valid_spkid_list)) + "\n")
stats_fn.write("Spekers " + " ".join(valid_spkid_list) + "\n")

for ft in feat_types_list:
   
    for st in stats_list:
        
        tofel_list = []
        pred_list = []
        
        for spkid in valid_spkid_list:
            tofel = spk2tofel_dict[spkid]
            pred = spk2stats_dict[spkid][ft][st]
            tofel_list.append(tofel)
            pred_list.append(pred)
        
        tofel_list = np.array(tofel_list)
        pred_list = np.array(pred_list)
        
        spc_res = stats.spearmanr(tofel_list, pred_list)
        pcc_res = stats.pearsonr(tofel_list, pred_list)
        
        img_fn = ft + "-" + st
        corr_info = ",".join([img_fn, str(spc_res.statistic), str(pcc_res.statistic)])
        stats_fn.write(corr_info + "\n")
        
        title = "-".join(corr_info.split())
        plot(tofel_list, pred_list, "tofel", "dtw", title, os.path.join(result_dir, img_fn + ".png"))
 
