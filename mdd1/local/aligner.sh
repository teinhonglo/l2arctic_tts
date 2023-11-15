#!/bin/bash

# Author: Fu-An Chao
# 	  this code is modified from Kaiqi Fu, JinJong Lin
stage=0

BACKEND=yourtts

l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
data_root=data-mdd
exp_root=exp/mdd
gpuid=0
test_sets=

set -euo pipefail
echo "$0 $@"

. ./path.sh
. ./utils/parse_options.sh

echo $test_sets

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

if [ $stage -le 0 ]; then
    echo -e "${GREEN}Aligned MDD data $data_root/ to $exp_root/...  with a neural phonetic aligner (e.g., charsiu) ${NC}"
    
    for test_set in $test_sets; do
        # aligned with original waveform files
        CUDA_VISIBLE_DEVICES=$gpuid \
            python local/aligner.py --data_dir $data_root/$test_set \
                                    --output_dir $exp_root/$test_set/align_org \
                                    --wavscp_fn wav.scp
        
        # aligned with reference waveform files
        CUDA_VISIBLE_DEVICES=$gpuid \
            python local/aligner.py --data_dir $data_root/$test_set \
                                    --output_dir $exp_root/$test_set/align_ref \
                                    --wavscp_fn wav_ref.scp 
    done
fi

