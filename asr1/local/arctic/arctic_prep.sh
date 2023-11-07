#!/usr/bin/env bash
set -e

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Copyright 2023 RCPET@NTNU (Fu-An Chao)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


db=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <spk> <data_dir>"
    exit 1
fi

# check speaker
available_spks=(
    "slt" "clb" "bdl" "rms" "jmk" "awb" "ksp"
)
if ! $(echo ${available_spks[*]} | grep -q ${spk}); then
    echo "Specified speaker ${spk} is not available."
    echo "Available speakers: ${available_spks[*]}"
    exit 1
fi

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
phn_text=${data_dir}/phn_text
wrd_text_ori=${data_dir}/wrd_text.ori
wrd_text=${data_dir}/wrd_text
transcript_phn_text=${data_dir}/transcript_phn_text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${phn_text} ] && rm ${phn_text}
[ -e ${wrd_text} ] && rm ${wrd_text}
[ -e ${transcript_phn_text} ] && rm ${transcript_phn_text}

# make scp
echo "making wav.scp"
find ${db} -name "arctic*.wav" -follow | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> ${scp}
done

# make phn_text (NOTE: arctic has 39 phone set + ax + pau)
echo "making phn_text, transcript_phn_text"
find ${db}/lab -name "arctic*.lab" -follow | sort | while read -r filename; do
    phn_list=""
    while read line; do
        phn=$(echo ${line} | cut -d " " -f 3)
        [ "${phn}" == "pau" ] && phn="sil"   # reduce  pau -> sil
        [ "${phn}" == "ax" ] && phn="ah"     # reduce   ax -> ah
        [ "${phn}" == "ssil" ] && phn="sil"  # reduce ssil -> sil (typo?)
        phn_list="$phn_list $phn"
    done < <(tail -n +2 $filename)
    echo ${spk}_$(basename "${filename}" .lab)${phn_list} >> ${phn_text}
done
cp ${phn_text} ${transcript_phn_text}

# make wrd_text
echo "making wrd_text"
raw_text=${db}/etc/txt.done.data
ids=$(sed < ${raw_text} -e "s/^( /${spk}_/g" -e "s/ )$//g" | cut -d " " -f 1)
#sentences=$(sed < ${raw_text} -e "s/^( //g" -e "s/ )$//g" -e "s/\"//g" | tr '[:lower:]' '[:upper:]' | cut -d " " -f 2-)
sentences_ori=$(sed < ${raw_text} -e "s/^( //g" -e "s/ )$//g" -e "s/\"//g" | cut -d " " -f 2-)
sentences=$(sed < ${raw_text} -e "s/^( //g" -e "s/ )$//g" -e "s/\"//g" -e "s/[,.]//g" | tr '[:upper:]' '[:lower:]' | cut -d " " -f 2-)

paste -d " " <(echo "${ids}") <(echo "${sentences_ori}") > ${wrd_text_ori}
paste -d " " <(echo "${ids}") <(echo "${sentences}") > ${wrd_text}
