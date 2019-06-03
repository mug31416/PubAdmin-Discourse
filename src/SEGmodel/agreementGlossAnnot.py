#!/usr/bin/env python3
import sys, argparse, os, pickle, json
sys.path.append('.')

import numpy as np

from sklearn.metrics import cohen_kappa_score

from fullTextIndex.textProcForIndex import *
from utils.spacyAnnot import *
from utils.misc import *
from utils.glossAnnotMarch2019 import *

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

tokExtr=TokenExtractor(STOP_WORDS)

SENT_QTY = 4

ERROR_FILE_NAME_PREF='error_annot'
ERROR_OWNERS= 'error_too_many_annot.json'


from utils.misc import *
from SEGmodel.commonForModeling import *

LABEL_MAP = {
  'ALL' : 1,
  'COMMENT_DISCUSSION' : 1,
  'NO_CHANGE_IN_RULE' : 1,
  'CHANGE_IN_RULE' : 1,
  'MULTI': {'COMMENT_DISCUSSION' : 1, 'NO_CHANGE_IN_RULE' : 2, 'CHANGE_IN_RULE' : 3 }
}

parser = argparse.ArgumentParser(description='GLOSS evaluation')

parser.add_argument('--glossJSON',
                    required=True, type=str,
                    help='A gloss JSON file')
parser.add_argument('--sectTitleTrainJSONL',
                    required=True, type=str,
                    help='train SPACY JSONL with title annotations, e.g. sectitle_train.jsonl')
parser.add_argument('--docSetJSON',
                    required=True, type=str,
                    help='document set JSON file')
parser.add_argument('--doubleAnnoList',
                    required=True, type=str,
                    help='A list of doubly annotated documents')
parser.add_argument('--label',
                    required=True, type=str,
                    choices=['COMMENT_DISCUSSION', 'CHANGE_IN_RULE','NO_CHANGE_IN_RULE','MULTI', 'ALL'],
                    help='Annotation type to evaluate')
parser.add_argument('--examples',
                    action = 'store_true',
                    help='Add examples of annotations')


args = parser.parse_args()


doubleAnnoList = []
with open(args.doubleAnnoList,'r') as f:
  for line in enumerate(f):
    doubleAnnoList.append(line[1].strip())

print('Double annotated docs:')
print(doubleAnnoList)

doubleAnnoSet = set(doubleAnnoList)

allAnnots = [[], []]


for docketId, doc_name, sectProps, docJSON in readGlossAnnots(tokExtr, args.glossJSON, args.sectTitleTrainJSONL, args.docSetJSON):

  print('!!!', len(sectProps))

  if not doc_name in doubleAnnoSet:
    print(f'Ignoring document {doc_name} b/c it does not have 0-2 annotators')
    continue
  else:
    print(f'Proceeeding to process document {doc_name} b/c it may have 0-2 annotators')

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

    if len(annotByOwner) > 2:
      with open(ERROR_OWNERS, 'w') as f:
        json.dump(docJSON, f)
      raise Exception('Invalid config: more than two annotators for document: ' + doc_name + ' check file: ' + ERROR_OWNERS)


    for sid in range(sectSentQty):
      start, end, sent_txt = oneSectProp.sectSents[sid]

      sentAnnot = [0, 0]

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

            oid = ownerId[owner]
            curVal = sentAnnot[oid]
            sentAnnot[oid] = max(replVal,curVal)

            if args.examples:
              print("@@@@EXAMPLE ",owner, atype, sent_txt, replVal)

      for i in range(2):
        allAnnots[i].append(sentAnnot[i])

      print('##', sentAnnot)


qty = len(allAnnots[0])
annot0 = np.array(allAnnots[0])
annot1 = np.array(allAnnots[1])

conf = confusion_matrix(annot0,annot1,labels=[0,1,2,3])

print(conf)



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