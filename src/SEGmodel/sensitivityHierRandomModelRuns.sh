#!/usr/bin/env bash

export F=2001
export H=101


export mt="rand_hier_mlp"
export mt1="mlp"
export mt2="cond_mlp"

date > log.summary_${mt}.txt
echo ${mt} >> log.summary_${mt}.txt

for feat in wlf_f4bert_head wlf_head bert_head wlf_bert_head f7bert_head; do

    for l in CHANGE_IN_RULE NO_CHANGE_IN_RULE; do

        python -u trainModel.py --train_file ../../data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/${feat}_${F}_${H} --label ${l} --use_nodseq_model --filter_all --two_stage --use_prior --model_file ${feat}_${F}_${H}/${mt2}_${feat}_${l} --first_stage_model_file ${feat}_${F}_${H}/${mt1}_${feat}_ALL 2>&1|tee log.${mt}_${feat}_${l}.txt ;

    done;

    echo NO_CHANGE_IN_RULE ${feat} >> log.summary_${mt}.txt ;
    tail -n 13 log.${mt}_${feat}_NO_CHANGE_IN_RULE.txt >> log.summary_${mt}.txt ;

    echo CHANGE_IN_RULE ${feat} >> log.summary_${mt}.txt ;
    tail -n 13 log.${mt}_${feat}_CHANGE_IN_RULE.txt >> log.summary_${mt}.txt ;

done




