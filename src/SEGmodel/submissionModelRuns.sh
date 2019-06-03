#!/usr/bin/env bash

export F=2001
export H=101


# SVM on different labels, WLF

#date > log.summary.txt

#for feat in wlf_head bert_head wlf_bert_head; do

    #for l in ALL  CHANGE_IN_RULE NO_CHANGE_IN_RULE; do

        #python -u trainCrfModel.py --train_file ../../data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/${feat}_${F}_${H} --label ${l} --use_nodseq_model --model_file ${feat}_${F}_${H}/svm_${feat}_${l} 2>&1|tee log.svm_${feat}_${l}.txt ;

    #done;

    #tail -n 13 log.svm_${feat}_ALL.txt >> log.summary.txt ;
    #tail -n 13 log.svm_${feat}_NO_CHANGE_IN_RULE.txt >> log.summary.txt ;
    #tail -n 13 log.svm_${feat}_CHANGE_IN_RULE.txt >> log.summary.txt ;

#done


# SVM on different labels, WLF

export mt="mlp"

date > log.summary_${mt}.txt
echo ${mt} >> log.summary_${mt}.txt

for feat in wlf_f4bert_head wlf_head bert_head wlf_bert_head f7bert_head; do

    for l in ALL  CHANGE_IN_RULE NO_CHANGE_IN_RULE; do

        python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/${feat}_${F}_${H} --label ${l} --use_nodseq_model --model_file ${feat}_${F}_${H}/${mt}_${feat}_${l} 2>&1|tee log.${mt}_${feat}_${l}.txt ;

    done;

    echo ALL ${feat} >> log.summary_${mt}.txt ;
    tail -n 13 log.${mt}_${feat}_ALL.txt >> log.summary_${mt}.txt ;

    echo NO_CHANGE_IN_RULE ${feat} >> log.summary_${mt}.txt ;
    tail -n 13 log.${mt}_${feat}_NO_CHANGE_IN_RULE.txt >> log.summary_${mt}.txt ;

    echo CHANGE_IN_RULE ${feat} >> log.summary_${mt}.txt ;
    tail -n 13 log.${mt}_${feat}_CHANGE_IN_RULE.txt >> log.summary_${mt}.txt ;

done



# CRF on different labels, WLF

#for l in ALL  CHANGE_IN_RULE NO_CHANGE_IN_RULE; do

    #python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/wlf_head_${F}_${H} --label ${l}  --model_file wlf_head_${F}_${H}/crf_${l} 2>&1|tee log.head_wlf_crf_${l}.txt ;

#done

#tail -n 5 log.head_wlf_svm_ALL.txt
#tail -n 5 log.head_wlf_svm_NO_CHANGE_IN_RULE.txt
#tail -n 5 log.head_wlf_svm_CHANGE_IN_RULE.txt


