#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# yourtts (pip install resemblyzer)
BACKEND=yourtts

# script
stage=0
gpuid=0
l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
data_root=data
exp_root=exp/secs
test_sets=all_16k # test_set
test_conds="yrtts yrtts_spkemb"
#test_conds="yrtts"
spkids="HKK,YDCK,YKWK,HJK,BWC,LXC,TXHC,NCC,YBAA,SKA,ABA,ZHAA,EBVS,NJS,ERMS,MBMPS,RRBI,TNI,ASI,SVBI,HQTV,PNV,TLV,THV"
score_opts=

# decode_options is used in Whisper model's transcribe method
#decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 10, fp16: False}"

. ./path.sh
. ./utils/parse_options.sh


if [ $stage -le 0 ]; then 
    for test_set in $test_sets; do
        
        if [ ! -d $exp_root/$test_set ]; then
            mkdir -p $exp_root/$test_set;
        fi
        
        for test_cond in $test_conds; do
            data_dir=data/$test_set
            data_dir2=${data_dir}_${test_cond}
            output_path=$exp_root/$test_set/cosine_sim-`basename $data_dir2`.txt
        
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python local/compute_spkemb_sim.py --data_dir $data_dir --spkids $spkids \
                                                   --data_dir2 $data_dir2 > $output_path;
        done
    done
fi
