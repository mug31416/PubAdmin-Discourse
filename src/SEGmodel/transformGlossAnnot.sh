#!/usr/bin/env bash


python SEGmodel/transformGlossAnnot.py --docSetJSON /Users/anna/Research/Docket/RegDocket/docketSampling/finalruleSampling/finalRulesDev1DocketDict.json --glossJSON /Users/anna/Research/Docket/RegDocket/annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0430.json --outFile /Users/anna/Research/Docket/RegDocket/data4Modeling/SecPassageClassification/dev1Dockets_2019-0430.jsonl --withContext

python SEGmodel/transformGlossAnnot.py --docSetJSON /Users/anna/Research/Docket/RegDocket/docketSampling/finalruleSampling/finalRulesDev2DocketDict.json --glossJSON /Users/anna/Research/Docket/RegDocket/annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0430.json --outFile /Users/anna/Research/Docket/RegDocket/data4Modeling/SecPassageClassification/dev2Dockets_2019-0430.jsonl --withContext

python SEGmodel/transformGlossAnnot.py --docSetJSON /Users/anna/Research/Docket/RegDocket/docketSampling/finalruleSampling/finalRulesTestDocketDict.json --glossJSON /Users/anna/Research/Docket/RegDocket/annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0430.json --outFile /Users/anna/Research/Docket/RegDocket/data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl --withContext


cat /Users/anna/Research/Docket/RegDocket/data4Modeling/SecPassageClassification/dev1Dockets_2019-0430.jsonl /Users/anna/Research/Docket/RegDocket/data4Modeling/SecPassageClassification/dev2Dockets_2019-0430.jsonl > /Users/anna/Research/Docket/RegDocket/data4Modeling/SecPassageClassification/trainDockets_2019-0430.jsonl
