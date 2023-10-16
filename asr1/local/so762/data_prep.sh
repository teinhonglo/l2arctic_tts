#!/bin/bash

stage=0

so762_dir="/share/corpus/speechocean762"
data_kaldi='data/so762'
corpus_prefix=so762_
age_limit=15

. ./path.sh
. ./utils/parse_options.sh

GREEN='\033[0;32m' # green
NC='\033[0m' # no color

if [ $stage -le 0 ]; then
    echo -e "${GREEN}kaldi data preparation ...${NC}"

    [ ! -d $data_kaldi ] && mkdir -p $data_kaldi
    
    for data_set in train test; do
        dest_dir=$data_kaldi/${corpus_prefix}$data_set
        utils/copy_data_dir.sh $so762_dir/$data_set $dest_dir
        
        cp -r $so762_dir/$data_set/spk2age $dest_dir/
        sed -i "s:WAVE:${so762_dir}/WAVE:g" $dest_dir/wav.scp
        awk '{for(i=2; i<=NF; i++) $i=tolower($i);print}' $dest_dir/text > $dest_dir/temp.txt
        mv $dest_dir/temp.txt $dest_dir/text
    done

    utils/combine_data.sh --extra_files spk2age $data_kaldi/${corpus_prefix}all $data_kaldi/${corpus_prefix}train $data_kaldi/${corpus_prefix}test
fi

if [ $stage -le 1 ]; then
    echo -e "${GREEN}create dataset under {age_limit} ...${NC}"
    src_all_dir=$data_kaldi/${corpus_prefix}all
    dest_all_dir=$data_kaldi/${corpus_prefix}all_u${age_limit}
    
    awk -v limit="$age_limit" '$2 <= limit {print $1}' $src_all_dir/spk2age > $src_all_dir/spklist.u${age_limit}
    utils/subset_data_dir.sh --spk-list $src_all_dir/spklist.u${age_limit} $src_all_dir $dest_all_dir
    cp $src_all_dir/spk2age $dest_all_dir
    
    utils/fix_data_dir.sh --spk_extra_files spk2age $dest_all_dir
fi
