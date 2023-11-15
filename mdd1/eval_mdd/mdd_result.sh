#!/bin/bash
# modified from https:/github.com/cageyoko/CTC-Attention-Mispronunciation/blob/master/egs/attention_aug/result/mdd_result.sh

if [ $# -ne 3 ]; then
    echo "Usage: $0 <human-seq> <ref> <hyp>"
    echo "  human-seq : human perceived results "
    echo "        ref : canonical phone sequences "
    exit 0
fi

human_seq=$1
ref=$2
hyp=$3

# step 0, filter & sort, ensure human_seq, ref contain sames utterances with hyp
eval_mdd/utils/filter_scp.pl -f 1 $hyp $human_seq | sort -nk1 > human_seq
eval_mdd/utils/filter_scp.pl -f 1 $hyp $ref | sort -nk1 > ref
sort -nk1 $hyp > hyp

# step 1
# note : sequence only have 39 phoneme, no sil
align-text ark:ref  ark:human_seq ark,t:- | eval_mdd/utils/wer_per_utt_details.pl > ref_human_detail
align-text ark:human_seq  ark:hyp ark,t:- | eval_mdd/utils/wer_per_utt_details.pl > human_our_detail
align-text ark:ref  ark:hyp ark,t:- | eval_mdd/utils/wer_per_utt_details.pl > ref_our_detail
python eval_mdd/utils/ins_del_sub_cor_analysis.py

# step 2
compute-wer --text --mode=present ark:human_seq ark:hyp || exit 1;

rm ref_human_detail human_our_detail ref_our_detail
rm human_seq ref hyp
