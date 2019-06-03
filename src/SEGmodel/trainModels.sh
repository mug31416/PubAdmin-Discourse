#!/usr/bin/env bash

export F=2001
export H=101


# Testing file load routines

#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/t1.jsonl --test_file ../../data4Modeling/SecPassageClassification/t2.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/tmp_head_${F}_${H}  --use_nodseq_model --label ALL --model_file tmp_rc/svm 2>&1|tee log.head_wlf_svm.rc

#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/t1.jsonl --test_file ../../data4Modeling/SecPassageClassification/t2.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/tmp_head_${F}_${H}   --label ALL --model_file tmp_rc/crf 2>&1|tee log.head_wlf_crf.rc


#python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/dev1Dockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/dev2Dockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/wlf_tf_head_${F}_${H} --use_tfidf --use_nodseq_model --label ALL --model_file wlf_tf_head_${F}_${H}/svm_ALL 2>&1|tee log.head_wlf_tf_svm_ALL.txt



python -u trainModels.py --train_file ../../data4Modeling/SecPassageClassification/dev1Dockets_2019-0430.jsonl --test_file ../../data4Modeling/SecPassageClassification/dev2Dockets_2019-0430.jsonl --head_feat_qty $H --feat_qty $F --feat_cache_dir ../../data4Modeling/SecPassageClassification/wlf_head_${F}_${H} --label CHANGE_IN_RULE --use_nodseq_model  2>&1|tee log.head_wlf_svm_ALL.txt




