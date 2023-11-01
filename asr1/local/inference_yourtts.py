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

parser.add_argument("--lang",
                    default=None,
                    type=str)

parser.add_argument("--data_dir",
                    default="data/all_16k",
                    type=str)

parser.add_argument("--output_dir",
                    default="data/yourtts_test",
                    type=str)

parser.add_argument("--model_path",
                    default="tts_models/multilingual/multi-dataset/your_tts",
                    type=str)

parser.add_argument("--spk_embed_type",
                    default="utt",
                    choices=["all", "utt", "male", "female"],
                    type=str)

parser.add_argument("--download",
                    default="true",
                    choices=["true", "false"],
                    type=str)

args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
output_wav_dir = os.path.join(output_dir, "wavs")
model_path = args.model_path
spk_embed_type = args.spk_embed_type
download = True if args.download == "true" else False

if not download:
    config_path = os.path.join(os.path.dirname(model_path), "config.json")

print("output dir: {output_dir}".format(output_dir=output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(output_wav_dir)
    
    files = ["text", "utt2spk", "spk2utt"]
    for fname in files:
        shutil.copy2(os.path.join(data_dir, fname), output_dir)
    
else:
    print("{output_dir} had already existed".format(output_dir=output_dir))
    exit(0)

src_wavscp = os.path.join(data_dir, "wav.scp")
src_text = os.path.join(data_dir, "text")
src_spk2utt = os.path.join(data_dir, "spk2utt")

uttid_list = []
src_wavscp_dict = {}
src_text_dict = {}
src_spk2utt_dict = {}

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

with open(src_spk2utt, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        spkid = info[0]
        src_spk2utt_dict[spkid] = info[1:]

tgt_wavscp = os.path.join(output_dir, "wav.scp")
tgt_wavscp_fn = open(tgt_wavscp, "w")

if download:
    tts = TTS(model_name=model_path, progress_bar=True, gpu=True)
else:
    tts = TTS(model_path=model_path, config_path=config_path, progress_bar=True, gpu=True)

# 合併該語者的所有句子，產生 spk.wav
if spk_embed_type == "all":
    from pydub import AudioSegment
    for spkid, uttids in src_spk2utt_dict.items():
        sounds = []
        output_wav_path = os.path.join(output_wav_dir, spkid + ".wav")
        
        for uttid in uttids:
            wav_path = src_wavscp_dict[uttid]
            src_wavscp_dict[uttid] = output_wav_path
            sounds.append(AudioSegment.from_wav(wav_path))
            
        combined_sounds = sounds[0]
        for sound in sounds[1:]:
            combined_sounds += sound
        
        combined_sounds.export(output_wav_path)
elif spk_embed_type == "utt":
    pass
elif spk_embed_type in ["male", "female"]:
    tts_spkid = tts.speakers[0] if spk_embed_type == "female" else tts.speakers[3]
else:
    pass

for uttid in tqdm(uttid_list):
    wav_path = src_wavscp_dict[uttid]
    text = src_text_dict[uttid]
    
    output_wav_path = os.path.join(output_wav_dir, uttid + ".wav")

    if spk_embed_type in ["male", "female"]:
        tts.tts_to_file(text, speaker=tts_spkid, language=lang, file_path=output_wav_path)
    else:
        tts.tts_to_file(text, speaker_wav=wav_path, language=lang, file_path=output_wav_path)
    
    tgt_wavscp_fn.write("{uttid} {output_wav_path}\n".format(uttid=uttid, output_wav_path=output_wav_path))
