#!/usr/bin/env bash

export F=2001
export H=101

# WLF
#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/dev1Dockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/dev2Dockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/wlf_head_${F}_${H} --label ALL --use_nodseq_model 2>&1|tee log.wlf_head_d.txt

#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/wlf_head_${F}_${H} --label ALL --use_nodseq_model 2>&1|tee log.wlf_head_t.txt

#WLF + BERT
#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/dev1Dockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/dev2Dockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/wlf_bert_head_${F}_${H} --use_bert --use_head_bert --label ALL --use_nodseq_model 2>&1|tee log.wlf_bert_head_d.txt

#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/wlf_bert_head_${F}_${H} --use_bert --use_head_bert --label ALL --use_nodseq_model 2>&1|tee log.wlf_bert_head_t.txt

#BERT
#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/dev1Dockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/dev2Dockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/bert_${F}_${H} --use_bert --no_wordfeat --label ALL --use_nodseq_model 2>&1|tee log.bert_head_d.txt

#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/bert_${F}_${H} --use_bert --no_wordfeat --label ALL --use_nodseq_model 2>&1|tee log.bert_head_t.txt


bertModelArg=" --bert_model_path $HOME/TextCollect/Regulations.gov/Dockets/finetuned_bert/pytorch_model_6.bin "

# Finetuned BERT (7 iterations)
#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/f7bert_head_${F}_${H} --use_bert --use_head_bert --label ALL --use_nodseq_model --no_wordfeat $bertModelArg 2>&1|tee log.f7bert_head_t.txt

# Finetuned BERT (7 iter) + WLF
python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/wlf_f7bert_head_${F}_${H} --use_bert --use_head_bert --label CHANGE_IN_RULE --use_nodseq_model $bertModelArg 2>&1|tee log.wlf_f7bert_head_t.txt
