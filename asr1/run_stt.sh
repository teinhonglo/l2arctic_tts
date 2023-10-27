#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# whisperx
BACKEND=whisperx
whisper_tag=tiny    # whisper model tag, e.g., tiny, small, medium, large, etc
language=en

# script
stage=0
stop_stage=100
gpuid=0
l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
data_root=data
exp_root=exp/stt
test_sets="all" # test_sets，可更改成這次需 decode 的 data_set
score_opts=

# decode_options is used in Whisper model's transcribe method
#decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 10, fp16: False}"

echo "$0 $@"

. ./path.sh
. ./utils/parse_options.sh


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    ./local/data_prep.sh --stage 0 \
                        --l2arctic_dir $l2arctic_dir \
                        --data_kaldi $data_root
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for test_set in $test_sets; do
        
        data_dir=$data_root/$test_set
        output_dir="$exp_root/whisper-$whisper_tag"/$test_set
        
        if [ -f $output_dir/RESULTS.md ]; then
            continue
        fi
        
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python local/recog_whisperx.py --data_dir $data_dir \
                                           --model_tag $whisper_tag \
                                           --language $language \
                                           --output_dir $output_dir
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        output_dir="$exp_root/whisper-$whisper_tag"/$test_set
        
        echo $output_dir        
        sclite \
            ${score_opts} \
                -r "$output_dir/ref" trn \
                -h "$output_dir/hyp" trn \
                -i rm -o all stdout > "${output_dir}/result.txt"
        
        echo ""
        echo "WER"
        grep -e Avg -e SPKR -m 2 "${output_dir}/result.txt" | tee ${output_dir}/RESULTS.md
        
        sclite \
            ${score_opts} \
                -r "$output_dir/ref_norm" trn \
                -h "$output_dir/hyp_norm" trn \
                -i rm -o all stdout > "${output_dir}/result.norm.txt"
        
        echo ""
        echo "WER (Normalized)"
        grep -e Avg -e SPKR -m 2 "${output_dir}/result.norm.txt" | tee -a ${output_dir}/RESULTS.md
        
        sclite \
            ${score_opts} \
                -r "$output_dir/ref_normc" trn \
                -h "$output_dir/hyp_normc" trn \
                -i rm -o all stdout > "${output_dir}/result.normc.txt"
        
        echo ""
        echo "CER (Normalized)"
        grep -e Avg -e SPKR -m 2 "${output_dir}/result.normc.txt" | tee -a ${output_dir}/RESULTS.md
        
        echo "======================================================="
    done
fi
