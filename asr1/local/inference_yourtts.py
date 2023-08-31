import os
import json
import soundfile
from tqdm import tqdm
import numpy as np
import sys
import shutil
import argparse

from TTS.api import TTS

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data/all_16k",
                    type=str)

parser.add_argument("--output_dir",
                    default="data/yourtts_test",
                    type=str)

parser.add_argument("--model_path",
                    default="tts_models/multilingual/multi-dataset/your_tts",
                    type=str)

args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
output_wav_dir = os.path.join(output_dir, "wavs")
model_path = args.model_path

print("output dir: {output_dir}".format(output_dir=output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(output_wav_dir)
    
    files = ["text", "utt2spk", "spk2utt"]
    for fname in files:
        shutil.copy2(os.path.join(data_dir, fname), output_dir)
    
else:
    print("{output_dir} is already existed".format(output_dir=output_dir))
    exit(0)

src_wavscp = os.path.join(data_dir, "wav.scp")
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

tgt_wavscp = os.path.join(output_dir, "wav.scp")
tgt_wavscp_fn = open(tgt_wavscp, "w")

tts = TTS(model_name=model_path, progress_bar=True, gpu=True)

for uttid in tqdm(uttid_list):
    wav_path = src_wavscp_dict[uttid]
    text = src_text_dict[uttid]
    
    output_wav_path = os.path.join(output_wav_dir, os.path.basename(wav_path))
    tts.tts_to_file(text, speaker_wav=wav_path, language="en", file_path=output_wav_path)
    
    tgt_wavscp_fn.write("{uttid} {output_wav_path}\n".format(uttid=uttid, output_wav_path=output_wav_path))


