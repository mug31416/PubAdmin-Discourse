#!/usr/bin/env bash

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesTestDocketDict.json --label ALL 2>&1|tee log.kappa.test.all_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev1DocketDict.json --label ALL 2>&1|tee log.kappa.dev1.all_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev2DocketDict.json --label ALL 2>&1|tee log.kappa.dev2.all_2019-0419

# reduced set of annotations, ALL

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesTestDocketDict.json --label ALL 2>&1|tee log.kappa.test.all.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev1DocketDict.json --label ALL 2>&1|tee log.kappa.dev1.all.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev2DocketDict.json --label ALL 2>&1|tee log.kappa.dev2.all.subset_2019-0419


# reduced set of annotations, MULTI

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesTestDocketDict.json --label MULTI 2>&1|tee log.kappa.test.multi.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev1DocketDict.json --label MULTI 2>&1|tee log.kappa.dev1.multi.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev2DocketDict.json --label MULTI 2>&1|tee log.kappa.dev2.multi.subset_2019-0419


# reduced set of annotations, COMMENT_DISCUSSION

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesTestDocketDict.json --label COMMENT_DISCUSSION 2>&1|tee log.kappa.test.comment_discussion.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev1DocketDict.json --label COMMENT_DISCUSSION 2>&1|tee log.kappa.dev1.comment_discussion.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev2DocketDict.json --label COMMENT_DISCUSSION 2>&1|tee log.kappa.dev2.comment_discussion.subset_2019-0419

# reduced set of annotations, NO_CHANGE_IN_RULE

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesTestDocketDict.json --label NO_CHANGE_IN_RULE 2>&1|tee log.kappa.test.no_change.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev1DocketDict.json --label NO_CHANGE_IN_RULE 2>&1|tee log.kappa.dev1.no_change.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev2DocketDict.json --label NO_CHANGE_IN_RULE 2>&1|tee log.kappa.dev2.no_change.subset_2019-0419

# reduced set of annotations, CHANGE_IN_RULE

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesTestDocketDict.json --label CHANGE_IN_RULE 2>&1|tee log.kappa.test.change.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev1DocketDict.json --label CHANGE_IN_RULE 2>&1|tee log.kappa.dev1.change.subset_2019-0419

time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev2DocketDict.json --label CHANGE_IN_RULE 2>&1|tee log.kappa.dev2.change.subset_2019-0419
