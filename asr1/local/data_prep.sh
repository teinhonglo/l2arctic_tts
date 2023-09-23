#!/bin/bash

stage=0

l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
data_kaldi='data'

. ./path.sh
. ./utils/parse_options.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

if [ $stage -le 0 ]; then
    echo -e "${GREEN}kaldi data preparation ...${NC}"

    [ ! -d $data_kaldi ] && mkdir -p $data_kaldi
    python local/l2arctic_prep.py --l2_path=$l2arctic_dir --save_path=${data_kaldi}/train_l2 || exit 1
    python local/l2arctic_prep.py --l2_path=$l2arctic_dir --save_path=${data_kaldi}/dev || exit 1
    python local/l2arctic_prep.py --l2_path=$l2arctic_dir --save_path=${data_kaldi}/test || exit 1
fi

if [ $stage -le 1 ]; then
    echo -e "${GREEN}create word-level targets ...${NC}"
    for data in train_l2 dev test; do
        cp ${data_kaldi}/$data/wrd_text ${data_kaldi}/$data/text
        awk -F"_" '{print $1}' ${data_kaldi}/$data/text > ${data_kaldi}/$data/spklist
        awk -F" " '{print $1}' ${data_kaldi}/$data/text > ${data_kaldi}/$data/uttlist
        paste -d" " ${data_kaldi}/$data/uttlist ${data_kaldi}/$data/spklist > ${data_kaldi}/$data/utt2spk
        rm -rf ${data_kaldi}/$data/uttlist ${data_kaldi}/$data/spklist
        utils/fix_data_dir.sh --utt_extra_files "wrd_text wav_sph.scp phn_text transcript_phn_text" ${data_kaldi}/$data
    done
fi


if [ $stage -le 2 ]; then
    echo -e "${GREEN}Combine all data ...${NC}"
    utils/combine_data.sh --extra_files "wrd_text wav_sph.scp phn_text transcript_phn_text" ${data_kaldi}/all ${data_kaldi}/train_l2 ${data_kaldi}/dev ${data_kaldi}/test
fi
