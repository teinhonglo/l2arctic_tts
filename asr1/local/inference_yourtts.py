import os
import uuid 
import subprocess
import json
import soundfile
from tqdm import tqdm
import numpy as np
import sys
import shutil
import argparse

from TTS.api import TTS

def voice_clean(speaker_wav, out_filename):
    lowpassfilter=denoise=trim=loudness=True
        
    if lowpassfilter:
        lowpass_highpass="lowpass=8000,highpass=75," 
    else:
        lowpass_highpass=""

    if trim:
        # better to remove silence in beginning and end for microphone
        trim_silence="areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
    else:
        trim_silence=""
            
    try: 
        #we will use newer ffmpeg as that has after denoise filter
        shell_command = f"./ffmpeg -y -i {speaker_wav} -af {lowpass_highpass}{trim_silence} {out_filename}".split(" ")
    
        command_result = subprocess.run([item for item in shell_command])
        speaker_wav=out_filename
    except subprocess.CalledProcessError:
        # There was an error - command exited with non-zero code
        print("Error: failed filtering, use original microphone input")
    
    return speaker_wav

# ArgumentParser
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

parser.add_argument("--spk_embed_type",
                    default="utt",
                    choices=["all", "utt", "male", "female"],
                    type=str)

parser.add_argument("--download",
                    default="true",
                    choices=["true", "false"],
                    type=str)

parser.add_argument("--voice_cleanup",
                    default="false",
                    choices=["true", "false"],
                    type=str)

args = parser.parse_args()

# args
data_dir = args.data_dir
output_dir = args.output_dir
output_wav_dir = os.path.join(output_dir, "wavs")
model_path = args.model_path
spk_embed_type = args.spk_embed_type
download = True if args.download == "true" else False
voice_cleanup = True if args.voice_cleanup == "true" else False

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

# Data Preparation
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


# TTS
if download:
    tts = TTS(model_name=model_path, progress_bar=True, gpu=True)
else:
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    tts = TTS(model_path=model_path, config_path=config_path, progress_bar=True, gpu=True)


for uttid in tqdm(uttid_list):
    speaker_wav = src_wavscp_dict[uttid]
    text = src_text_dict[uttid]
    
    output_wav_path = os.path.join(output_wav_dir, uttid + ".wav")

    if spk_embed_type in ["male", "female"]:
        tts.tts_to_file(text, speaker=tts_spkid, language="en", file_path=output_wav_path)
    else:
        if voice_cleanup:
            speaker_wav = voice_clean(speaker_wav, output_wav_path.split(".wav")[0] + "_cln.wav")
            tts.tts_to_file(text, speaker_wav=speaker_wav, language="en", file_path=output_wav_path)
            
            shell_command = f"rm -rf {speaker_wav}".split(" ")
            command_result = subprocess.run([item for item in shell_command])
        else:
            tts.tts_to_file(text, speaker_wav=speaker_wav, language="en", file_path=output_wav_path)
            
    tgt_wavscp_fn.write("{uttid} {output_wav_path}\n".format(uttid=uttid, output_wav_path=output_wav_path))
