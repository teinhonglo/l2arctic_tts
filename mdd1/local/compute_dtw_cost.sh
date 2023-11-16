#!/bin/bash

# Author: Fu-An Chao
# 	  this code is modified from Kaiqi Fu, JinJong Lin

BACKEND=yourtts

l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
data_root='data-mdd'
exp_root=exp/mdd
align_dir=align_org
align_dir2=align_ref
feats="wav2vec2"
intervals="words,phones"
stage=0
stop_stage=10000
test_sets=

. ./path.sh
. ./utils/parse_options.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo -e "${GREEN}Compute DTW distance between $align_dir and $align_dir2 ${NC}"
    for test_set in $test_sets; do

        src_align_dir=$exp_root/$test_set/$align_dir
        tgt_align_dir=$exp_root/$test_set/$align_dir2   
        src_wavscp=$data_root/$test_set/wav.scp
        tgt_wavscp=$data_root/$test_set/wav_ref.scp
        output_root=$exp_root/$test_set/dtw_${feats}

        if [ ! -d $output_root/phones ]; then
            echo "$output_root/phones had already exsited."
            wait 5
            continue
        fi

        python local/compute_dtw_cost.py    --src_align_dir $src_align_dir \
                                            --tgt_align_dir $tgt_align_dir \
                                            --src_wavscp $src_wavscp \
                                            --tgt_wavscp $tgt_wavscp \
                                            --intervals $intervals \
                                            --feats $feats --output_root $output_root
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    valid_set=dev
    test_set=test
    python local/mispronunciation_dtw_evaluator.py --valid_transcript data-mdd/$valid_set/transcript_phn_text \
                                                    --valid_targets data-mdd/$valid_set/detection_targets \
                                                    --valid_dtw_folder exp/mdd/$valid_set/dtw_wav2vec2/phones \
                                                    --test_transcript data-mdd/$test_set/transcript_phn_text \
                                                    --test_targets data-mdd/$test_set/detection_targets \
                                                    --test_dtw_folder exp/mdd/$test_set/dtw_wav2vec2/phones \
                                                    --save_path exp/mdd/$test_set/dtw_${feats}/dtw.png
fi
