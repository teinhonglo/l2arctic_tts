import os
import json
from tqdm import tqdm
import sys
import shutil
from collections import defaultdict
import argparse
import torch

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tslearn.metrics import dtw, dtw_path
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, WhisperModel, WhisperFeatureExtractor

# global parameters, avoid reload in multiple times.
wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")
whisper_encoder = WhisperModel.from_pretrained("openai/whisper-small.en").encoder


def load_audio(file_path):
    return librosa.load(file_path, sr=16000)

def get_spectrogram(y, sr):
    S = librosa.feature.melspectrogram( y=y, sr=sr, n_fft=1024, 
                                        hop_length=256, n_mels=80, 
                                        window="hann", center=True, 
                                        fmin=80, fmax=7600)

    S_dB = librosa.power_to_db(S, ref=np.max)
    
    return S_dB                                      

def get_wav2vec2_features(y, sr):
    input_values = wav2vec2_feature_extractor(y, return_tensors="pt", sampling_rate=sr).input_values

    with torch.no_grad():
        outputs = wav2vec2_model(input_values)
        hidden_states = outputs.last_hidden_state[0]

    return hidden_states.numpy()

def get_whisper_features(y, sr):
    input_features = whisper_feature_extractor(y, return_tensors="pt", sampling_rate=sr).input_features[0]
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

def main(original_file, synthesis_file, output_dir, log_fn):
    original_fname = os.path.basename(original_file).split(".")[0]
    
    y1, sr1 = load_audio(original_file)
    y2, sr2 = load_audio(synthesis_file)
    spec1 = get_spectrogram(y1, sr1)
    spec2 = get_spectrogram(y2, sr2)
    w2v_vec1 = get_wav2vec2_features(y1, sr1)
    w2v_vec2 = get_wav2vec2_features(y2, sr2)
    #wsp_vec1 = get_whisper_features(y1, sr1)
    #wsp_vec2 = get_whisper_features(y2, sr2)
    
    # Compute and save DTW distances
    path_waveform, distance_waveform = compute_dtw_distance(y1, y2)
    path_spectrogram, distance_spectrogram = compute_dtw_distance(spec1.T, spec2.T)
    path_wav2vec2, distance_wav2vec2 = compute_dtw_distance(w2v_vec1, w2v_vec2)
    #path_whisper, distance_whisper = compute_dtw_distance(wsp_vec1, wsp_vec2)

    log_fn.write(f"DTW distance for waveform: {distance_waveform}\n")
    log_fn.write(f"DTW distance for spectrogram: {distance_spectrogram}\n")
    log_fn.write(f"DTW distance for wav2vec2: {distance_wav2vec2}\n")
    #log_fn.write(f"DTW distance for whisper: {distance_whisper}\n")

    # Plot and save DTW alignment paths
    plot_dtw_alignment(path_waveform, y1, y2, sr1, sr2, "DTW Alignment for Waveform", output_dir, original_fname + "-dtw_waveform.png")
    plot_dtw_alignment_2d(path_spectrogram, spec1, spec2, sr1, sr2, "DTW Alignment for Spectrogram", output_dir, original_fname + "-dtw_spectrogram.png")
    plot_dtw_alignment_2d(path_wav2vec2, w2v_vec1, w2v_vec2, sr1, sr2, "DTW Alignment for Wav2Vec2", output_dir, original_fname + "-dtw_wav2vec2.png")
    #plot_dtw_alignment_2d(path_whisper, wsp_vec1, wsp_vec2, sr1, sr2, "DTW Alignment for Whisper", output_dir, original_fname + "-dtw_whisper.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default="data/all_16k",
                        type=str)

    parser.add_argument("--data_dir2",
                        default="data/yourtts_test",
                        type=str)
    
    parser.add_argument("--exp_root",
                        default="exp/dtw",
                        type=str)

    parser.add_argument("--spkids",
                        default=None,
                        type=str)

    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir2 = args.data_dir2
    exp_root = args.exp_root
    spkids = args.spkids

    src_wavscp = os.path.join(data_dir, "wav.scp")
    src_utt2spk = os.path.join(data_dir, "utt2spk")
    src_utt2spk_dict = {}
    src_wavs_dict = {}

    tgt_wavscp = os.path.join(data_dir2, "wav.scp")
    tgt_wavs_dict = {}

    output_dir = os.path.join(exp_root, os.path.basename(data_dir), os.path.basename(data_dir2))
    img_output_dir = os.path.join(output_dir, "images")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(img_output_dir)

    uttid_list = []

    with open(src_wavscp, "r") as fn:
        for line in fn.readlines():
            uttid, wav_path = line.split()
            src_wavs_dict[uttid] = wav_path
            uttid_list.append(uttid)

    with open(src_utt2spk, "r") as fn:
        for line in fn.readlines():
            uttid, spkid = line.split()
            src_utt2spk_dict[uttid] = spkid

    with open(tgt_wavscp, "r") as fn:
        for line in fn.readlines():
            uttid, wav_path = line.split()
            tgt_wavs_dict[uttid] = wav_path

    if spkids is None:
        spkid_list = list(set(list(src_utt2spk_dict.values())))
    else:
        spkid_list = spkids.split(",")

    
    results_log_fn = os.path.join(output_dir, "results.log")
    
    ignored_uttid = set()
    
    if os.path.isfile(results_log_fn):
    
        with open(results_log_fn, "r") as fn:
            for line in fn.readlines():
                if "Utterance" in line:
                    uttid = line.split()[-1]
                    ignored_uttid.add(uttid)
        
        log_fn = open(results_log_fn, "a")
    else:
        log_fn = open(results_log_fn, "w")
    
    for uttid in tqdm(uttid_list):
        if uttid in ignored_uttid: continue
        
        spkid = src_utt2spk_dict[uttid]
        src_wav_path = src_wavs_dict[uttid]
        tgt_wav_path = tgt_wavs_dict[uttid]
        
        log_fn.write(f"Utterance ID: {uttid}\n")
        main(src_wav_path, tgt_wav_path, img_output_dir, log_fn)

    log_fn.close()
