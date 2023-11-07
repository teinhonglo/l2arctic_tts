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
voice_cleanup=false

# SpeechMos
batch_size=4

# script
gpuid=0
stage=0
stop_stage=1
data_root=data
test_sets="all_16k" # test_set
score_opts=
lang= # "en" when using a multilingual model

# decode_options is used in Whisper model's transcribe method
#decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 10, fp16: False}"

. ./path.sh
. ./utils/parse_options.sh


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        output_dir=${data_dir}${model_affix}
        
        
        if [ ! -z $lang ] ; then
            CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --lang $lang \
                                              --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --download "$download" \
                                              --voice_cleanup $voice_cleanup
        else
            CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --download "$download" \
                                              --voice_cleanup $voice_cleanup
	fi
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/eval_pseudomos.py $output_dir/wav.scp --outdir $output_dir --batch_size $batch_size 
    done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for test_set in $test_sets; do
        data_dir=${data_root}/${test_set}   
        output_dir=${data_dir}${model_affix}_spkemb
        
        if [ ! -z $lang ] ; then
            CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --lang $lang \
                                              --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --spk_embed_type "all" \
                                              --download "$download" \
                                              --voice_cleanup $voice_cleanup
        else
            CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --spk_embed_type "all" \
                                              --download "$download" \
                                              --voice_cleanup $voice_cleanup
	fi 
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/eval_pseudomos.py $output_dir/wav.scp --outdir $output_dir --batch_size $batch_size 
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set    
        output_dir=${data_dir}${model_affix}_spkm
        
        if [ ! -z $lang ] ; then
            CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --lang $lang \
                                              --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --spk_embed_type "male" \
                                              --download "$download" \
                                              --voice_cleanup $voice_cleanup
        else
            CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --spk_embed_type "male" \
                                              --download "$download" \
                                              --voice_cleanup $voice_cleanup
	fi 
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/eval_pseudomos.py $output_dir/wav.scp --outdir $output_dir --batch_size $batch_size
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set    
        output_dir=${data_dir}${model_affix}_spkfm
        
        if [ ! -z $lang ] ; then
            CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --lang $lang \
                                              --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --spk_embed_type "female" \
                                              --download "$download" \
                                              --voice_cleanup $voice_cleanup
        else
            CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/inference_yourtts.py --data_dir $data_dir \
                                              --output_dir $output_dir \
                                              --model_path $model_path \
                                              --spk_embed_type "female" \
                                              --download "$download" \
                                              --voice_cleanup $voice_cleanup
	fi 
        
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/eval_pseudomos.py $output_dir/wav.scp --outdir $output_dir --batch_size $batch_size
    done
fi
