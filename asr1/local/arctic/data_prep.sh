#!/bin/bash


stage=-1

arctic_dir=/share/corpus/arctic
spks="slt bdl rms clb jmk awb ksp"
data_kaldi='data/arctic'
corpus_prefix=arctic_

. ./path.sh
. ./utils/parse_options.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

if [ ${stage} -le -1 ]; then
    for spk in $spks; do
        echo -e "${GREEN}download cmu arctic corpus for speaker ${spk} ...${NC}"
        local/arctic/data_download.sh ${arctic_dir} ${spk}
    	echo
    done
fi

if [ ${stage} -le 0 ]; then
    [ ! -d $data_kaldi ] && mkdir -p $data_kaldi
    for spk in $spks; do
        echo -e "${GREEN}kaldi data preparation for speaker $spk ...${NC}"
    	local/arctic/arctic_prep.sh ${arctic_dir}/cmu_us_${spk}_arctic ${spk} $data_kaldi/${spk}
	    echo
    done
fi

if [ ${stage} -le 1 ]; then
    echo -e "${GREEN}creating dataset for training ...${NC}"
    echo -e "${GREEN}---------------------------------${NC}"
    echo -e "${GREEN}train: slt, bdl, rms, clb${NC}"
    echo -e "${GREEN}dev: jmk${NC}"
    echo -e "${GREEN}test: awb${NC}"
    echo -e "${GREEN}---------------------------------${NC}"

    # train: slt, bdl, rms, clb, ksp
    for data_set in slt bdl rms clb ksp jmk awb; do
        rm -rf $data_kaldi/$data_set/{utt2spk, spk2utt}
        cp $data_kaldi/$data_set/wrd_text $data_kaldi/$data_set/text
        awk -v spkid=$data_set '{print $1" "spkid}' $data_kaldi/$data_set/text > $data_kaldi/$data_set/utt2spk
        utils/fix_data_dir.sh $data_kaldi/$data_set
        
    done 

    utils/combine_data.sh $data_kaldi/cmu_arctic_7spks $data_kaldi/{slt,bdl,rms,clb,ksp,jmk,awb}
    exit 0
fi
