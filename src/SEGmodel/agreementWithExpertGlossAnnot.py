#!/usr/bin/env python3
import sys, argparse, os, pickle, json
sys.path.append('.')

import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score

from fullTextIndex.textProcForIndex import *
from utils.spacyAnnot import *
from utils.misc import *
from utils.glossAnnotMarch2019 import *

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

tokExtr=TokenExtractor(STOP_WORDS)

SENT_QTY = 4

ERROR_FILE_NAME_PREF='error_annot'


from utils.misc import *
from SEGmodel.commonForModeling import *

LABEL_MAP = {
  'ALL' : 1,
  'COMMENT_DISCUSSION' : 1,
  'NO_CHANGE_IN_RULE' : 1,
  'CHANGE_IN_RULE' : 1,
  'MULTI': {'COMMENT_DISCUSSION' : 1, 'NO_CHANGE_IN_RULE' : 2, 'CHANGE_IN_RULE' : 3 }
}


parser = argparse.ArgumentParser(description='GLOSS annoation agreement with expert')

parser.add_argument('--glossJSON',
                    required=True, type=str,
                    help='A gloss JSON file')
parser.add_argument('--sectTitleTrainJSONL',
                    required=True, type=str,
                    help='train SPACY JSONL with title annotations, e.g. sectitle_train.jsonl')
parser.add_argument('--docSetJSON',
                    required=True, type=str,
                    help='document set JSON file')
parser.add_argument('--outname',
                    required=True, type=str,
                    help='output file name')
parser.add_argument('--label',
                    required=True, type=str,
                    choices=['COMMENT_DISCUSSION', 'CHANGE_IN_RULE','NO_CHANGE_IN_RULE','MULTI', 'ALL'],
                    help='Annotation type to evaluate')
parser.add_argument('--examples',
                    action = 'store_true',
                    help='Add examples of annotations')


args = parser.parse_args()


annotatorList = [ "5c5c75d0ee951f365779eaca",
                  "5c5c75f0ee951f365779eacb",
                  "5c5c760dee951f365779eacc",
                  "5c6c6d4580930b282483d698",
                  "5c5c7644ee951f365779eace",
                  "5c5c765aee951f365779eacf",
                  "5c5c7670ee951f365779ead0",
                  "5c5c7689ee951f365779ead1",
                  "5c6c6d6d80930b282483d699",
                  "5c5c76b8ee951f365779ead3",
                  "5c491ecbee951f365779e552"
                  ]

allAnnots = {}
allAnnots['document'] = []
allAnnots['section'] = []
allAnnots['sentence'] = []
for ann in annotatorList:
  allAnnots[ann] = []

for docketId, doc_name, sectProps, docJSON in readGlossAnnots(tokExtr, args.glossJSON,
                                                              args.sectTitleTrainJSONL,
                                                              args.docSetJSON, exploreAgreement=True):

  print('!!!', len(sectProps))


  for oneSectProp in sectProps:

    sectSentQty = len(oneSectProp.sectSents)

    print('@@ sentence # @@', sectSentQty)

    annotByOwner = dict()
    ownerId = dict()

    for ast, aend, owner, _ in oneSectProp.sectAnnot:

      if not owner in annotByOwner:
        ownerId[owner] = len(annotByOwner)
        annotByOwner[owner] = []

      annotByOwner[owner].append( (ast, aend) )


    print(doc_name + ' annotators: ' + str(ownerId.keys()))

    for sid in range(sectSentQty):
      start, end, sent_txt = oneSectProp.sectSents[sid]

      allAnnots['document'].append(doc_name)
      allAnnots['section'].append(oneSectProp.sectId)
      allAnnots['sentence'].append(sid)
      for ann in annotatorList:
        allAnnots[ann].append(0)

      for ast, aend, owner, atype in oneSectProp.sectAnnot:

        s1 = max(start, ast)
        s2 = min(end, aend)

        if args.label != "MULTI":
          replVal = LABEL_MAP[atype]

        if args.label != "ALL" and args.label != "MULTI":
          if atype != args.label:
            continue

        if args.label == "MULTI":
          replVal = LABEL_MAP["MULTI"][atype]

        if s1 < s2 and end > start:

          cov = float(s2 - s1) / (end - start)

          if cov >= MIN_ANNOT_OVERLAP:

            if owner in annotatorList:

              #print("#########",owner,allAnnots[owner])
              curVal = allAnnots[owner][-1]
              allAnnots[owner][-1] = max(replVal,curVal)

            if args.examples:
              print("@@@@EXAMPLE ",owner, atype, sent_txt, replVal)



df = pd.DataFrame.from_dict(allAnnots)
df.to_csv(args.outname, index=False)


if False:
  kappa = cohen_kappa_score(annot0, annot1)
  print('Kappa        : %s' % formatVal(kappa))
  print('N_all        : %s' % qty)

  ind0 = np.arange(qty)[np.logical_or(annot1 == 0, annot0 ==0)]
  kappa = cohen_kappa_score(annot0[ind0], annot1[ind0])
  print('Kappa (at least one is 0): %s' % formatVal(kappa))
  r = float(np.sum((annot0[ind0] == annot1[ind0])))/len(ind0)
  print('match ratio              : %g' % r)
  print('N_0                      : %d' % len(ind0))

  ind1 = np.arange(qty)[np.logical_or(annot1 == 1, annot0 ==1)]
  kappa = cohen_kappa_score(annot0[ind1], annot1[ind1])
  print('Kappa (at least one is 1): %s' % formatVal(kappa))
  r = float(np.sum((annot0[ind1] == annot1[ind1])))/len(ind1)
  print('match ratio              : %g' % r)
  print('N_1                      : %d' % len(ind1))

  if args.label == "MULTI":

    ind2 = np.arange(qty)[np.logical_or(annot1 == 2, annot0 == 2)]
    kappa = cohen_kappa_score(annot0[ind2], annot1[ind2])
    print('Kappa (at least one is 2): %s' % formatVal(kappa))
    r = float(np.sum((annot0[ind2] == annot1[ind2]))) / len(ind2)
    print('match ratio              : %g' % r)
    print('N_2                      : %d' % len(ind2))

    ind3 = np.arange(qty)[np.logical_or(annot1 == 3, annot0 == 3)]
    kappa = cohen_kappa_score(annot0[ind3], annot1[ind3])
    print('Kappa (at least one is 3): %s' % formatVal(kappa))
    r = float(np.sum((annot0[ind3] == annot1[ind3]))) / len(ind3)
    print('match ratio              : %g' % r)
    print('N_3                      : %d' % len(ind3))