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
stop_stage=1000
gpuid=0
l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
data_root=data
exp_root=exp/dtw
data_dir=all_16k # test_set
data_dir2=all_16k_yrtts
#test_conds="yrtts"
spkids="HKK,YDCK,YKWK,HJK,BWC,LXC,TXHC,NCC,YBAA,SKA,ABA,ZHAA,EBVS,NJS,ERMS,MBMPS,RRBI,TNI,ASI,SVBI,HQTV,PNV,TLV,THV"
score_opts=

# decode_options is used in Whisper model's transcribe method
#decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 10, fp16: False}"

. ./path.sh
. ./utils/parse_options.sh


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python local/compute_and_viz_dtw.py --data_dir $data_root/$data_dir \
                                        --data_dir2 $data_root/$data_dir2 \
                                        --exp_root $exp_root
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python local/proficiency_correlation.py --data_dir $data_root/$data_dir \
                                        --log_fn $exp_root/$data_dir/$data_dir2/results.log 
fi

