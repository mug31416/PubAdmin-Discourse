#!/usr/bin/env bash

# reduced set of annotations, MULTI

#time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesTestDocketDict.json --label MULTI --examples 2>&1|tee log.kappa.test.multi.subset.examples_2019-0426

#time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev1DocketDict.json --label MULTI --examples 2>&1|tee log.kappa.dev1.multi.subset.examples_2019-0426

#time python3 SEGmodel/agreementGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0419.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl --doubleAnnoList ../annotations/glossDocumentAnnotation_2019-xx/double_annotated_multi_list_2019-0419.txt --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev2DocketDict.json --label MULTI --examples 2>&1|tee log.kappa.dev2.multi.subset.examples_2019-0426

# Explore agreement with expert, MULTI

time python3 SEGmodel/agreementWithExpertGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0430.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl  --docSetJSON ../docketSampling/finalruleSampling/finalRulesTestDocketDict.json --outname testDocketAgreement.csv --label MULTI  2>&1|tee log.expert.kappa.test.multi_2019-0502

time python3 SEGmodel/agreementWithExpertGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0430.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl  --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev1DocketDict.json --outname dev1DocketAgreement.csv --label MULTI  2>&1|tee log.expert.kappa.dev1.multi_2019-0502

time python3 SEGmodel/agreementWithExpertGlossAnnot.py --glossJSON ../annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0430.json --sectTitleTrainJSONL ../data4Modeling/SecTitleClassification/sectitle_train.jsonl  --docSetJSON ../docketSampling/finalruleSampling/finalRulesDev2DocketDict.json --outname dev2DocketAgreement.csv --label MULTI  2>&1|tee log.expert.kappa.dev2.multi_2019-0502

