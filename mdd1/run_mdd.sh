#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# yourtts
BACKEND=yourtts
# corpus-related
l2arctic_dir="/share/corpus/l2arctic_release_v4.0"

# script
stage=0
stop_stage=100
gpuid=0
data_kaldi=data
data_root=data-mdd
exp_root=exp/mdd
test_sets="all train_l2 dev test" # test_sets，可更改成這次需 decode 的 data_set
score_opts=

echo "$0 $@"

. ./path.sh
. ./utils/parse_options.sh


if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
    ./local/data_prep.sh --stage -1 \
                        --l2arctic_dir $l2arctic_dir \
                        --data_kaldi $data_kaldi \
                        --data_mdd $data_root
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    ./local/aligner.sh  --data_root $data_root \
                        --exp_root $exp_root \
                        --gpuid $gpuid \
                        --test_sets "$test_sets"
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ./local/compute_dtw_cost.sh --data_root $data_root \
                                --exp_root $exp_root \
                                --align_dir align_org \
                                --align_dir2 align_ref \
                                --intervals "phones" \
                                --test_sets "test dev train_l2 all"
fi

# mdd
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    valid_set=dev
    test_set=test
    python local/mispronunciation_dtw_evaluator.py --valid_transcript data-mdd/$valid_set/transcript_phn_text \
                                                    --valid_targets data-mdd/$valid_set/detection_targets \
                                                    --valid_dtw_folder exp/mdd/$valid_set/dtw_wav2vec2/phones \
                                                    --test_transcript data-mdd/$test_set/transcript_phn_text \
                                                    --test_targets data-mdd/$test_set/detection_targets \
                                                    --test_dtw_folder exp/mdd/$test_set/dtw_wav2vec2/phones \
                                                    --save_path exp/mdd/$test_set/dtw_wav2vec2/dtw.png
fi
