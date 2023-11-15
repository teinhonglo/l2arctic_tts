import os
import json
from tqdm import tqdm
import sys
import shutil
from collections import defaultdict
import argparse
import torch
import textgrid

import numpy as np
import librosa
import matplotlib.pyplot as plt
from tslearn.metrics import dtw, dtw_path
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, WhisperModel, WhisperFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global parameters, avoid reload in multiple times.
wav2vec2_model = "facebook/wav2vec2-base-960h"
whisper_model = "openai/whisper-small.en"
wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_model)
wav2vec2_model = Wav2Vec2Model.from_pretrained(wav2vec2_model)
whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
whisper_encoder = WhisperModel.from_pretrained(whisper_model).encoder


def load_audio(file_path):
    return librosa.load(file_path, sr=16000)

def get_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, 
                                       hop_length=256, n_mels=80, 
                                       window="hann", center=True, 
                                       fmin=80, fmax=7600)

    #S_dB = librosa.power_to_db(S, ref=np.max)
    
    return S

def get_wav2vec2_features(y, sr):
    input_values = wav2vec2_feature_extractor(y, return_tensors="pt", sampling_rate=sr, padding=True).input_values

    with torch.no_grad():
        outputs = wav2vec2_model(input_values)
        hidden_states = outputs.last_hidden_state[0]

    return hidden_states.numpy()

def get_whisper_features(y, sr):
    input_features = whisper_feature_extractor(y, return_tensors="pt", sampling_rate=sr, padding=True).input_features[0]
    input_features = input_features.unsqueeze(0)

    with torch.no_grad():
        outputs = whisper_encoder(input_features)
        hidden_states = outputs.last_hidden_state[0]

    return hidden_states.numpy()

def compute_dtw_distance(x, y):
    path, score = dtw_path(x, y)
    return path, score

def plot_dtw_alignment(path, y1, y2, sr1, sr2, title, output_dir, filename):
    x_idx, y_idx = zip(*path)
    fig, ax = plt.subplot_mosaic("hSSS;hSSS;hSSS;.vvv")
    plt.subplots_adjust(hspace=0.5)

    ax['S'].plot(x_idx, y_idx)
    ax['S'].label_outer()
    ax['S'].set(title=title)
    
    librosa.display.waveshow(y=y1, sr=sr1, ax=ax['v'])
    ax['v'].label_outer()
    ax['v'].set(title="Original Speech")

    librosa.display.waveshow(y=y2, sr=sr2, ax=ax['h'], transpose=True)
    ax['h'].label_outer()
    ax['h'].set(title="Synthesis Speech")
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_dtw_alignment_2d(path, y1, y2, sr1, sr2, title, output_dir, filename):
    x_idx, y_idx = zip(*path)
    fig, ax = plt.subplot_mosaic("hSSS;hSSS;hSSS;.vvv")
    plt.subplots_adjust(hspace=0.5)

    ax['S'].plot(x_idx, y_idx)
    ax['S'].label_outer()
    ax['S'].set(title=title)
    
    librosa.display.specshow(y1, sr=sr1, ax=ax['v'])
    ax['v'].label_outer()
    ax['v'].set(title="Original Speech")

    y2_rotated_flipped = np.fliplr(y2.T)
    librosa.display.specshow(y2_rotated_flipped, sr=sr2, ax=ax['h'])
    ax['h'].label_outer()
    ax['h'].transAxes.inverted()
    ax['h'].set(title="Synthesis Speech")
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()    

def seq_dtw(src_wav_path, tgt_wav_path, src_intervals, tgt_intervals, feats, output_root, uttid, interval_list=["phones", "words"]):
    src_audio, src_sr = load_audio(src_wav_path)
    tgt_audio, tgt_sr = load_audio(tgt_wav_path)

    assert src_sr == tgt_sr

    # feature extraction
    if feats == "spectrogram":
        # spectrogram
        src_feats = get_spectrogram(src_audio, src_sr)
        tgt_feats = get_spectrogram(tgt_audio, tgt_sr)
        resolution = src_sr / 256  # Assuming hop_length=256

    elif feats == "wav2vec2":
        # wav2vec2
        src_feats = get_wav2vec2_features(src_audio, src_sr)
        tgt_feats = get_wav2vec2_features(tgt_audio, tgt_sr)
        resolution = src_audio.shape[0] / src_feats.shape[0]  # Approximation, adjust based on model specifics

    elif feats == "whisper":
        # whisper
        src_feats = get_whisper_features(src_audio, src_sr)
        tgt_feats = get_whisper_features(tgt_audio, tgt_sr)
        resolution = src_audio.shape[0] / src_feats.shape[0]  # Approximation, adjust based on model specifics

    else:
        # waveform
        src_feats, tgt_feats = src_audio, tgt_audio
        resolution = src_sr

    for interval in interval_list:
        output_dir = os.path.join(output_root, interval)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        log_fn = open(os.path.join(output_dir, uttid + ".log"), "w")
        #if len(src_intervals[interval]) != len(tgt_intervals[interval]): continue
        assert len(src_intervals[interval]) == len(tgt_intervals[interval])
        
        for (src_start, src_end, src_text), (tgt_start, tgt_end, tgt_text) in zip(src_intervals[interval], tgt_intervals[interval]):
            assert src_text == tgt_text
            # Sampling Points
            src_start_index = int(src_start * src_sr / resolution)
            src_end_index = int(src_end * src_sr / resolution)
            tgt_start_index = int(tgt_start * tgt_sr / resolution)
            tgt_end_index = int(tgt_end * tgt_sr / resolution)
        
            if src_end_index == src_start_index:
                src_end_index += 1
        
            if tgt_end_index == tgt_start_index:
                tgt_end_index += 1
        
            # Segmentation
            src_segment = src_feats[src_start_index:src_end_index]
            tgt_segment = tgt_feats[tgt_start_index:tgt_end_index]
        
            path, distance = compute_dtw_distance(src_segment, tgt_segment)
            log_fn.write(f"{src_text},{distance}\n")
        
        log_fn.close()


def extract_intervals_with_textgrid(textgrid_path, tier_names):
    tg = textgrid.TextGrid.fromFile(textgrid_path)

    intervals_dict = {}
    for tier_name in tier_names:
        tier = tg.getFirst(tier_name)
        intervals = [(interval.minTime, interval.maxTime, interval.mark) for interval in tier if interval.mark != "[SIL]"]
        intervals_dict[tier_name] = intervals

    return intervals_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_align_dir",
                        default="exp/mdd/all/align_org",
                        type=str)

    parser.add_argument("--tgt_align_dir",
                        default="exp/mdd/all/align_ref",
                        type=str)
    
    parser.add_argument("--src_wavscp",
                        default="data-mdd/all/wav.scp",
                        type=str)

    parser.add_argument("--tgt_wavscp",
                        default="data-mdd/all/wav_ref.scp",
                        type=str)

    parser.add_argument("--feats",
                        default="wav2vec2",
                        type=str)

    parser.add_argument("--output_root",
                        default="exp/mdd/all/dtw_wav2vec2",
                        type=str)                        
    
    parser.add_argument("--intervals",
                        default="words,phones",
                        type=str)                        

    args = parser.parse_args()

    src_align_dir = args.src_align_dir
    tgt_align_dir = args.tgt_align_dir
    src_wavscp = args.src_wavscp
    tgt_wavscp = args.tgt_wavscp
    feats = args.feats
    output_root = args.output_root
    
    interval_list = [itv for itv in args.intervals.split(",")] 

    src_wavs_dict = {}
    tgt_wavs_dict = {}
    
    uttid_list = []

    with open(src_wavscp, "r") as fn:
        for line in fn.readlines():
            uttid, wav_path = line.split()
            src_wavs_dict[uttid] = wav_path
            uttid_list.append(uttid)

    with open(tgt_wavscp, "r") as fn:
        for line in fn.readlines():
            uttid, wav_path = line.split()
            tgt_wavs_dict[uttid] = wav_path
    
    ignored_uttid = set()
    
    for uttid in tqdm(uttid_list):
        if uttid in ignored_uttid: continue
        
        src_wav_path = src_wavs_dict[uttid]
        tgt_wav_path = tgt_wavs_dict[uttid]
        # Get word_timestamp and phone_timestamp from TextGrid 
        src_textgrid_path = os.path.join(src_align_dir, uttid + ".TextGrid")
        tgt_textgrid_path = os.path.join(tgt_align_dir, uttid + ".TextGrid")
        src_intervals = extract_intervals_with_textgrid(src_textgrid_path, interval_list)
        tgt_intervals = extract_intervals_with_textgrid(tgt_textgrid_path, interval_list)

        seq_dtw(src_wav_path, tgt_wav_path, src_intervals, tgt_intervals, feats, output_root, uttid, interval_list)
    
