#!/usr/bin/env python3
'''
Ingest Gloss data and prepare analytical dataset
- split into sections
- TODO: split into sentences using legal text sentence tokenizer
- tokenize using spacy, with adjustments for eventual BERT
- probably remove URL and citations (?)
- create sentence labels, label confidence scores, number of annotators
  (to-do later additional work on annotator credibility from the test result set)
- add the section-level header data
- export as a JSONL
'''

import sys, argparse, os
import numpy as np
import json

sys.path.append('.')
from fullTextIndex.textProcForIndex import *
from utils.spacyAnnot import *
from utils.misc import *
from utils.glossAnnotMarch2019 import *
from SEGmodel.commonForModeling import *

tokExtr=TokenExtractor(STOP_WORDS)

SENT_QTY = 1


def to_int_list(arr):
  if len(arr.shape) == 1:
    return [int(x) for x in arr]
  elif len(arr.shape) == 2:
    return [[int(x) for x in e] for e in arr]

def main(argv):

  parser = argparse.ArgumentParser(description='GLOSS data transformation')

  parser.add_argument('--glossJSON',
                      required=False, type=str,
                      default="./annotations/glossDocumentAnnotation_2019-xx/procAnnotationsT12_2019-0430.json",
                      help='A gloss JSON file')
  parser.add_argument('--sectTitleJSONL',
                      required=False, type=str,
                      default="./data4Modeling/SecTitleClassification/sectitle_train.jsonl",
                      help='JSONL with title annotations, e.g. sectitle_train.jsonl')
  parser.add_argument('--docSetJSON',
                      required=False, type=str,
                      default="/./docketSampling/finalruleSampling/finalRulesTestDocketDict.json",
                      help='document set JSON file')
  parser.add_argument('--outFile',
                      required=False, type=str,
                      default="./data4Modeling/SecPassageClassification/testDockets.jsonl",
                      help='JSONL output file name')
  parser.add_argument('--withContext',
                      action='store_true', default=False, help='Include BERT context in the sentece')

  args = parser.parse_args(argv)
  print(args)

  with open(args.outFile, 'w') as tf:

    for docketId, doc_id, sectProps, _ in readGlossAnnots(tokExtr, args.glossJSON,
                                                          args.sectTitleJSONL,
                                                          args.docSetJSON):

      docContext = doc_id

      for oneSectProp in sectProps:

        sectSentQty = len(oneSectProp.sectSents)

        sentRelev = {}

        for at in ANNO_TYPES:
          sentRelev[at] = np.zeros(sectSentQty)

        sentRanges = []
        sentText = []

        secTitleRaw = oneSectProp.sectCompTitle if len(oneSectProp.sectCompTitle.strip())>0 else "PREAMBLE"

        secContext = secTitleRaw

        if args.withContext:
          secTitle = '[CLS] ' + secTitleRaw + ' [SEP] ' + docContext + ' [SEP]'

        else:
          secTitle = secTitleRaw

        docContext = doc_id + " " + secTitleRaw

        for sidStart in range(0, sectSentQty, SENT_QTY):

          sidEnd = min(sidStart + SENT_QTY, sectSentQty)

          _, _, sent_txt = oneSectProp.sectSents[sidStart:sidEnd][0]

          if args.withContext:
            sentText.append('[CLS] '+ sent_txt +' [SEP] ' + secContext + ' [SEP]')

          else:
            sentText.append(sent_txt)

          secContext = secTitleRaw + " " + sent_txt

          sentRanges.append( (sidStart, sidEnd))

        for ast, aend, _, atype in oneSectProp.sectAnnot:

          for sid in range(sectSentQty):

            start, end, _ = oneSectProp.sectSents[sid]

            s1 = max(start, ast)
            s2 = min(end, aend)

            if s1 < s2 and end > start:

              cov = float(s2 - s1) / (end - start)

              if cov >= MIN_ANNOT_OVERLAP:
                sentRelev[atype][sid] += 1

        for at in sentRelev:
          sentRelev[at]=to_int_list(sentRelev[at])

        obj = {"docketId": docketId,
               "doc_id": doc_id,
               "sect_id" : oneSectProp.sectId,
               "sect_offset" : oneSectProp.startOffset,
               "sent_text": sentText,
               "sent_labels": sentRelev,
               "section_name": secTitle,
               "section_label": oneSectProp.titleSectFlag,

               }

        #print(obj)

        objStr = json.dumps(obj)
        tf.write(objStr + '\n')


if __name__ == '__main__':
  main(sys.argv[1:])