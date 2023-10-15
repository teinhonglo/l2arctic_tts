import os
import traceback
import json
import soundfile
from tqdm import tqdm
import numpy as np
import sys
import shutil
import argparse

from utils.prompt_making import make_prompt
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

preload_models()

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data/all_16k",
                    type=str)

parser.add_argument("--output_dir",
                    default="data/all_16k_valle",
                    type=str)

parser.add_argument("--model_path",
                    default="tts_models/multilingual/multi-dataset/your_tts",
                    type=str)

parser.add_argument("--spk_embed_type",
                    default="utt",
                    choices=["all", "utt"],
                    type=str)

parser.add_argument("--language",
                    default="auto",
                    choices=["auto", "en", "zh", "ja"],
                    type=str)

parser.add_argument("--accent",
                    default="no-accent",
                    choices=["no-accent", "English", "中文", "日本語", "Mix"],
                    type=str)

parser.add_argument("--temperature",
                    default=1.0,
                    type=float)

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
language = args.language
accent = args.accent
temperature = args.temperature
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
    
    tgt_wavscp = os.path.join(output_dir, "wav.scp")
    tgt_wavscp_fn = open(tgt_wavscp, "w")
    
else:
    print("[WARNING] {output_dir} is already existed".format(output_dir=output_dir))
    tgt_wavscp = os.path.join(output_dir, "wav.scp")
    tgt_wavscp_fn = open(tgt_wavscp, "a")
    #exit(0)

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
else:
    pass

for uttid in tqdm(uttid_list):
    wav_path = src_wavscp_dict[uttid]
    text = src_text_dict[uttid]
    
    output_wav_path = os.path.join(output_wav_dir, uttid + ".wav")
     
    """
    if spk_embed_type == "utt":
        ### Use given transcript
        make_prompt(name=uttid, audio_prompt_path=wav_path, transcript=text)
    else:
        ### Alternatively, use whisper
        make_prompt(name=uttid, audio_prompt_path=wav_path)

    audio_array = generate_audio(text, prompt=uttid, language=language, accent=accent)
    
    write_wav(output_wav_path, SAMPLE_RATE, audio_array)
    tgt_wavscp_fn.write("{uttid} {output_wav_path}\n".format(uttid=uttid, output_wav_path=output_wav_path))
    """

    success = False  # 標記變量，用來檢查操作是否成功
    while not success and not os.path.isfile(output_wav_path):
        try:
            if spk_embed_type == "utt":
                ### Use given transcript
                make_prompt(name=uttid, audio_prompt_path=wav_path, transcript=text)
            else:
                ### Alternatively, use whisper
                make_prompt(name=uttid, audio_prompt_path=wav_path)

            audio_array = generate_audio(text, prompt=uttid, language=language, accent=accent)
            number_secs = audio_array.shape[0] / SAMPLE_RATE
            
            if number_secs <= 1.0:
                continue
            
            write_wav(output_wav_path, SAMPLE_RATE, audio_array)
            success = True  # 如果所有操作都成功，則設置成功標記為True以退出循環

        except Exception as e:
            print(f"An error occurred: {e}")
            print(traceback.format_exc())  # 印出錯誤的堆疊追蹤以幫助除錯
            # 可選：如果需要，可以在這裡添加一些延遲以避免立即重試
            import time
            time.sleep(1)
    
    tgt_wavscp_fn.write(f"{uttid} {output_wav_path}\n")
