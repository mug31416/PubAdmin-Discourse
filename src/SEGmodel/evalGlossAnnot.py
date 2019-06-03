#!/usr/bin/env python3
import sys, argparse, os, pickle
sys.path.append('.')

import numpy as np
import pickle

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

parser = argparse.ArgumentParser(description='GLOSS evaluation')

parser.add_argument('--glossJSON',
                    required=True, type=str,
                    help='A gloss JSON file')
parser.add_argument('--modelFile',
                    required=True, type=str,
                    help='A binary classifier passage classifier (for comment types)')
parser.add_argument('--sectTitleTrainJSONL',
                    required=True, type=str,
                    help='train SPACY JSONL with title annotations, e.g. sectitle_train.jsonl')
parser.add_argument('--docSetJSON',
                    required=True, type=str,
                    help='document set JSON file')
parser.add_argument('--avgType',
                    default='binary', type=str,
                    help='f1/recall/precision averaging type')


args = parser.parse_args()

modelFileName = args.modelFile
model = pickle.load(open(modelFileName, 'rb'))
print('Loaded model from file: ', modelFileName)

avgType = args.avgType

error_file_n = 0

allRelev = []
allPred = []

for docketId, doc_id, sectProps, _ in readGlossAnnots(tokExtr, args.glossJSON, args.sectTitleTrainJSONL, args.docSetJSON):

  for oneSectProp in sectProps:

    sectSentQty = len(oneSectProp.sectSents)
    sentRelev = np.zeros(sectSentQty)
    sentPreds = np.zeros(sectSentQty)

    feats = []
    sentRanges = []

    for sidStart in range(0, sectSentQty, SENT_QTY):
      sidEnd = min(sidStart + SENT_QTY, sectSentQty)
      passText = ' '.join([e[2] for e in oneSectProp.sectSents[sidStart:sidEnd]])
      sentRanges.append( (sidStart, sidEnd))

      featTxt = getDocPassTextForModel(tokExtr,
                                       True, oneSectProp.titleSectFlag,
                                       True, oneSectProp.sectCompTitle,
                                       True, passText)
      feats.append(featTxt)

    preds = model.predict(feats)

    for predId in range(len(preds)):
      sidStart, sidEnd = sentRanges[predId]
      for sid in range(sidStart, sidEnd):
        sentPreds[sid] = preds[predId]


    for ast, aend, _ in oneSectProp.sectAnnot:

      for sid in range(sectSentQty):
        start, end, _ = oneSectProp.sectSents[sid]
        s1 = max(start, ast)
        s2 = min(end, aend)
        if s1 < s2 and end > start:
          cov = float(s2 - s1) / (end - start)
          if cov >= MIN_ANNOT_OVERLAP:
            sentRelev[sid] += 1


    print('Ground-truth sentence relevance:')
    print(sentRelev)
    print('Predicted sentence relevance')
    print(sentPreds)

    allRelev.extend(list( (sentRelev > 0).astype(int)))
    allPred.extend(list(sentPreds))

    print('Relevant sentences:')
    for sid in range(sectSentQty):
      if sentRelev[sid] > 0:
        print(sentRelev[sid], oneSectProp.sectSents[sid][2])


totQty = len(allRelev)
for i in range(totQty):
  print(str(allRelev[i]) + " " + str(allPred[i]))

f1 = f1_score(allRelev, allPred, average=avgType)
recall = recall_score(allRelev, allPred, average=avgType)
prec = precision_score(allRelev, allPred, average=avgType)

print('F1        : %s' % formatVal(f1))
print('Recall    : %s' % formatVal(recall))
print('Precision : %s' % formatVal(prec))
