#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# yourtts
BACKEND=yourtts
model_path="tts_models/multilingual/multi-dataset/your_tts"
model_affix=_yrtts
download=true

# script
gpuid=0
stage=0
stop_stage=0
l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
data_root=data
test_sets="all_16k" # test_set
score_opts=

# decode_options is used in Whisper model's transcribe method
#decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 10, fp16: False}"

. ./path.sh
. ./utils/parse_options.sh


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        output_dir=${data_dir}${model_affix}
        
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --download "$download"
    done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set    
        output_dir=${data_dir}${model_affix}_spkemb
        
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --spk_embed_type "all" \
                                              --download "$download"
    done
fi
