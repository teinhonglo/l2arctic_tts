#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

nj=4
train_cmd=run.pl
audio_format=wav
fs=16k               # Sampling rate.
stage=0
data_root=data
test_sets=
resample_dir=

. ./path.sh
. ./utils/parse_options.sh



if [ $stage -le 0 ]; then
    echo "Stage 3: Format wav.scp: data/ -> ${resample_dir}"
    
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        
        if [ -z $resample_dir ]; then
            resample_dir=${data_dir}_${fs}
        fi
        
        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.
        
        utils/copy_data_dir.sh --validate_opts --non-print $data_dir "${resample_dir}"
        rm -f ${resample_dir}/{segments,wav.scp,reco2file_and_channel,reco2dur}

        # Copy reference text files if there is more than 1 reference
        _opts=
        if [ -e ${data_dir}/segments ]; then
            # "segments" is used for splitting wav files which are written in "wav".scp
            # into utterances. The file format of segments:
            #   <segment_id> <record_id> <start_time> <end_time>
            #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
            # Where the time is written in seconds.
            _opts+="--segments ${data_dir}/segments "
        fi
        # shellcheck disable=SC2086
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
            "$data_dir/wav.scp" "${resample_dir}" 
    done
fi
