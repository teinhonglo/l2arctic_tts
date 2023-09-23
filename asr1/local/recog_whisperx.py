import os
import json
import soundfile
from tqdm import tqdm
import numpy as np
import sys
from whisper.normalizers import BasicTextNormalizer
from whisper.normalizers import EnglishTextNormalizer
import whisperx
import string
import jiwer
import gc

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data/all",
                    type=str)

parser.add_argument("--output_dir",
                    default="whisper-tiny/all",
                    type=str) 

parser.add_argument("--model_tag",
                    default="medium",
                    type=str)
 
parser.add_argument("--device",
                    default="cuda",
                    type=str)

parser.add_argument("--language",
                    default="en",
                    type=str) 

parser.add_argument("--compute_type",
                    default="int8",
                    type=str) 

parser.add_argument("--batch_size",
                    default=16,
                    type=int) 

parser.add_argument("--beam_size",
                    default=10,
                    type=int) 

parser.add_argument("--condition_on_previous_text", action="store_true", help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")

args = parser.parse_args()

data_dir = args.data_dir
model_tag = args.model_tag
language = args.language
condition_on_previous_text = args.condition_on_previous_text
device = args.device
compute_type = args.compute_type
batch_size = args.batch_size
beam_size = args.beam_size
output_dir = args.output_dir

print("output dir: {output_dir}".format(output_dir=output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

decode_options = {
                      "suppress_tokens": None, 
                      "beam_size": beam_size,
                      "condition_on_previous_text": condition_on_previous_text
                 }

model = whisperx.load_model(model_tag, device, compute_type=compute_type, language=language, asr_options=decode_options)
normalizer = BasicTextNormalizer()
normalizer_en = EnglishTextNormalizer()

wavscp_dict = {}
text_dict = {}
utt_list = []
all_info = {}


with open(data_dir + "/wav.scp", "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        wavscp_dict[info[0]] = info[1]
        utt_list.append(info[0])

with open(data_dir + "/text", "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_dict[info[0]] = " ".join(info[1:])


for i, uttid in tqdm(enumerate(utt_list)):
    audio_file = wavscp_dict[uttid]
    ref = text_dict[uttid]
    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    text = [ result['segments'][i]['text'] for i in range(len(result['segments'])) ]
    hyp = " ".join(text)    
    #hyp = " ".join(normalizer(hyp).split())

    ref_norm = normalizer_en(ref)
    hyp_norm = normalizer_en(hyp)
    
    all_info[uttid] = { 
                        "ref": ref, "ref_norm": ref_norm, 
                        "hyp": hyp, "hyp_norm": hyp_norm
                      }
    if i % 500 == 0 and i != 0:
        print(uttid, all_info[uttid])


print(output_dir)

with open(output_dir + "/all.json", "w") as fn:
    json.dump(all_info, fn, indent=4, ensure_ascii=False)

# human transcription
ref_fn = open(output_dir + "/ref", "w")
# human transcription (norm)
ref_norm_fn = open(output_dir + "/ref_norm", "w")
# human transcription (normc)
ref_normc_fn = open(output_dir + "/ref_normc", "w")
# STT transcription
hyp_fn = open(output_dir + "/hyp", "w")
# STT transcription (norm)
hyp_norm_fn = open(output_dir + "/hyp_norm", "w")
# STT transcription (normc)
hyp_normc_fn = open(output_dir + "/hyp_normc", "w")

def charlize(utter):
    utter_nospace = "".join(utter.split())
    utter_char = " ".join(list(utter_nospace))
    return utter_char

# human transcription
for uttid in utt_list:
    if uttid in all_info:
        # reference
        ref_fn.write("{utter} ({uttid})\n".format(utter=all_info[uttid]["ref"], uttid=uttid))
        ref_norm_fn.write("{utter} ({uttid})\n".format(utter=all_info[uttid]["ref_norm"], uttid=uttid))
        ref_normc_fn.write("{utter} ({uttid})\n".format(utter=charlize(all_info[uttid]["ref_norm"]), uttid=uttid))
        # hypothesis
        hyp_fn.write("{utter} ({uttid})\n".format(utter=all_info[uttid]["hyp"], uttid=uttid))
        hyp_norm_fn.write("{utter} ({uttid})\n".format(utter=all_info[uttid]["hyp_norm"], uttid=uttid))
        hyp_normc_fn.write("{utter} ({uttid})\n".format(utter=charlize(all_info[uttid]["hyp_norm"]), uttid=uttid))


ref_fn.close()
ref_norm_fn.close()
ref_normc_fn.close()
hyp_fn.close()
hyp_norm_fn.close()
hyp_normc_fn.close()

