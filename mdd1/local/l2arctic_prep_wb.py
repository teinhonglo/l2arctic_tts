import glob
import os
import string
import textgrid
import re
import argparse
parser = argparse.ArgumentParser(description="Prepare L2 data")
parser.add_argument("--l2_path",default="/home/ljh/fkq/L2-Arctic",help="l2-Arctic path")
parser.add_argument("--save_path",default="./data/l2_train",help="l2-Arctic path")

args = parser.parse_args()

path = args.l2_path+"/*/annotation/*.TextGrid"
#   Spanish\Vietnamese\Hindi\Mandarin\Korean\Arabic
train_spk = ["EBVS","ERMS","HQTV","PNV","ASI","RRBI","BWC","LXC","HJK","HKK","ABA","SKA"]
dev_spk = ["MBMPS","THV","SVBI","NCC","YDCK","YBAA"]
test_spk = ["NJS","TLV","TNI","TXHC","YKWK","ZHAA"] 
#load_error_file = ["YDCK/annotation/arctic_a0209.TextGrid",
#                  "YDCK/annotation/arctic_a0272.TextGrid"]
load_error_file = []
wav_lst = glob.glob(path)
tmp_path = args.save_path
type_ = args.save_path.split("/")[-1].split("_")
w1 = open(tmp_path+"/transcript_phn_text_wb",'w+')
#w5 = open(tmp_path+"/detection_labels",'w+') # NOTE: detection labels

def extract_intervals_with_textgrid(textgrid_path, tier_names):
    tg = textgrid.TextGrid.fromFile(textgrid_path)

    intervals_dict = {}
    for tier_name in tier_names:
        tier = tg.getFirst(tier_name)
        intervals = [(interval.minTime, interval.maxTime, interval.mark) for interval in tier if interval.mark not in [ "[SIL]", "" ] ]
        intervals_dict[tier_name] = intervals

    return intervals_dict


def phone_intervals_normalize(phone_intervals):
    # normalize
    for i, (minTime, maxTime, phn) in enumerate(phone_intervals):
        if len(phn.split(",")) > 1:
            # transcript (correct pronunciation)
            phn = phn.split(",")[0]
        
        if(phn == "sp" or phn == "SIL" or phn == " " or phn == "spn" ):
            phn = "sil"
        else:
            phn = phn.strip(" ")
            if(phn == "ERR" or phn == "err"):
                phn = "err"
            elif(phn == "ER)"):
                phn = "er"
            elif(phn == "AX" or phn == "ax" or phn == "AH)"):
                phn = "ah"
            elif(phn == "V``"):
                phn = "v"
            elif(phn == "W`"):
                phn = "w"
        
        phone_intervals[i] = (minTime, maxTime, phn)

    # remove silence
    phone_intervals_nosil = []
    for i, (minTime, maxTime, phn) in enumerate(phone_intervals):
        if phn == "sil": continue
        
        phone_intervals_nosil.append((minTime, maxTime, phn))

    return phone_intervals_nosil

     
def align_phones_to_words(phone_intervals, word_intervals, tolerance=0.003):
    aligned_phones = []

    for word_start, word_end, word_text in word_intervals:
        word_phones = []

        for phone_start, phone_end, phone_text in phone_intervals:
            # 检查音素是否在单词的时间范围内
            if phone_start >= word_start - tolerance and phone_end <= word_end + tolerance:
                word_phones.append(phone_text)

        aligned_phones.append(tuple(word_phones))

    return aligned_phones


for phn_path in wav_lst:
    spk_id = phn_path.split("/")[-3]
    utt_id = spk_id + "_" + phn_path.split("/")[-1][:-9]
    
    intervals = extract_intervals_with_textgrid(phn_path, ["phones", "words"])
    phone_intervals, word_intervals = intervals["phones"], intervals["words"]
    phone_intervals = phone_intervals_normalize(phone_intervals)
 
    aligned_phones = align_phones_to_words(phone_intervals, word_intervals)

    filtered_data = [tup for tup in aligned_phones if len(tup) > 0]
    aligned_phones = filtered_data

    w1.write(f"{utt_id} {aligned_phones}\n")


w1.close()
