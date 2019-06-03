#!/usr/bin/env python3
import sys
sys.path.append('.')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

import numpy as np
import numpy.random as rnd

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import pickle

import eli5


from utils.spacyAnnot import *
from fullTextIndex.textProcForIndex import *
from models.commonForModeling import *
from utils.io import *
from utils.misc import *

from collections import namedtuple

Elem = namedtuple('Elem', 'binaryType, type, titleSectFlag, title, text')
Elem.__new__.__defaults__ = (None,) * len(Elem._fields)

def readJsonData(inFileName):
  res = []
  for e in readJson(inFileName):
    res.append(
      Elem(binaryType=e[BINARY_TYPE],
           type=e[TYPE],
           titleSectFlag=e[TITLE_SECT_FLAG],
           title=e[TITLE_FIELD],
           text=e[SPACY_TEXT_FIELD])
      )

  return res


def readData(tokExtr, inFileName, modelConfig):
  entries = []
  labels = []
  origText = []

  print('Model features:', modelConfig.modelFeatures)

  useTitle = ('title' in modelConfig.modelFeatures)
  useText = ('text' in modelConfig.modelFeatures)
  useTitleSectFlag = ('titleSectFlag' in modelConfig.modelFeatures)

  for e in readJsonData(inFileName):
    entries.append(getDocPassTextForModel(tokExtr,
                                          useTitleSectFlag, e.titleSectFlag,
                                          useTitle, e.title,
                                          useText, e.text))
    origText.append(e.text)

    if modelConfig.isMultiClass:
      labels.append(SPAN_ANNOT_TYPE_CODE[e.type])
    else:
      labels.append(int(e.binaryType))

  return np.array(labels), np.array(entries), np.array(origText)

class RandomChoice:

  def fit(self, _, labels):
    self.classQtys = dict()

    self.totQty = 0

    for lab in labels:
      self.totQty += 1
      if lab not in self.classQtys:
        self.classQtys[lab] = 0

      self.classQtys[lab] += 1

    self.labList = list(self.classQtys.keys())

  def predict(self, dt):
    dtQty = dt.shape[0]
    res = np.zeros(dtQty)

    for i in range(dtQty):
      r = rnd.randint(0, self.totQty) + 1
      s = 0
      pred = None
      for lab, qty in self.classQtys.items():
        s += qty
        if s >= r:
          pred = lab
          break

      if pred is None:
        raise Exception('Bug, we should not  reach this point, sum=%d random=%d!' % (s, r))

      res[i] = pred

    return res

CV_FOLDS=5

NUM_TREES=50
NUM_GBM_ITER=500
MAX_ITER=2000
ERR_SAMPLE_QTY=0

if len(sys.argv) != 4 and len(sys.argv) != 5:
  print('Usage: <input data JSON> <output model file> <model config JSON> <optional: result table>')
  sys.exit(1)


inFileName = sys.argv[1]
outFileName = sys.argv[2]
modelConfig = dictToClass('ModelConfig', readJson(sys.argv[3]))
outTableFileName = None
if len(sys.argv) == 5:
  outTableFileName = sys.argv[4]

print('Model config:')
print(modelConfig)

isMultiClass = modelConfig.isMultiClass == 1
useStopWords = modelConfig.useStopWords == 1
modelType = modelConfig.modelType
configName = modelConfig.configName

tokExtr = TokenExtractor(STOP_WORDS) if useStopWords else TokenExtractor([])

allLabels, allEntries, allOrigText = readData(tokExtr, inFileName, modelConfig)

print('Read %d entries' % len(allEntries))

seedAll(0)

def createClassifier(modelType):
  if modelType == 'linearSVM':
    clf = LinearSVC(class_weight='balanced')
  elif modelType == "randomForests":
    clf = RandomForestClassifier(n_jobs = -1,
                                 class_weight="balanced",
                                 n_estimators = NUM_TREES,
                                  verbose = 0)
  elif modelType == "logisticRegression":
    clf = LogisticRegression(penalty='l1',
                             n_jobs = -1,
                             verbose = 0,
                             max_iter = MAX_ITER,
                             class_weight='balanced')
  elif modelType == 'random':
    clf = RandomChoice()
  else:
    raise Exception('Unsupported model type: %s' % modelType)

  return clf


# We use bigrams
vectorizer = CountVectorizer(ngram_range=modelConfig.ngramRange)

LABEL_LIST = list(set(allLabels))
LABEL_LIST.sort()

TOP_K_MODEL_FEATURES = 25 * (len(LABEL_LIST))
TOP_K_MODEL_PRED_FEATURES = 5 * (len(LABEL_LIST))

oneVsAllF1Avg=np.zeros(len(LABEL_LIST))
oneVsAllRecallAvg=np.zeros(len(LABEL_LIST))
oneVsAllPrecAvg=np.zeros(len(LABEL_LIST))

multiClassF1Avg=0
multiClassRecallAvg=0
multiClassPrecAvg=0

confMatrix = None

confSampleDict = dict()

for iter in range(CV_FOLDS):
  clfCV = createClassifier(modelType)
  pipeCV = Pipeline([('vectorizer', vectorizer),
                   ('tfidf', TfidfTransformer()),
                   ('clf', clfCV)])


  trainEntries, testEntries, \
  trainLabels, testLabels, \
  _, testOrigText = train_test_split(allEntries, allLabels, allOrigText)

  pipeCV.fit(trainEntries, trainLabels)
  pred = pipeCV.predict(testEntries)

  for i in range(len(pred)):
    if pred[i] != testLabels[i]:
      key = (testLabels[i], pred[i])
      if key not in confSampleDict:
        confSampleDict[key] = []
      predExplan = eli5.format_as_text(eli5.explain_prediction(clfCV, testEntries[i],
                                                  top=TOP_K_MODEL_PRED_FEATURES,
                                                  vec=vectorizer))
      confSampleDict[key].append( (testOrigText[i], testEntries[i], pred[i], predExplan) )

  cm = confusion_matrix(testLabels, pred) / (CV_FOLDS * len(testLabels))

  if confMatrix is None:
    confMatrix = cm
  else:
    confMatrix += cm

  f1 = f1_score(testLabels, pred, average=AVG_TYPE)
  multiClassF1Avg += f1 / CV_FOLDS
  recall = recall_score(testLabels, pred, average=AVG_TYPE)
  multiClassRecallAvg += recall / CV_FOLDS
  prec = precision_score(testLabels, pred, average=AVG_TYPE)
  multiClassPrecAvg +=  prec / CV_FOLDS

  print('iter=%d' % iter)
  print('Multi-class averaging %s' % AVG_TYPE)

  print('F1        : %g' % f1)
  print('Recall    : %g' % recall)
  print('Precision : %g' % prec)

  for posLabel in LABEL_LIST:
    predBinary = (pred == posLabel).astype('int')
    labelBinary = (testLabels == posLabel).astype('int')

    f1score = f1_score(labelBinary, predBinary,
                       pos_label=1,  # We all convert to one for the current class and to zero to all others
                       average='binary')
    oneVsAllF1Avg[posLabel] += f1score / CV_FOLDS

    recall = recall_score(labelBinary, predBinary,
                       pos_label=1,  # We all convert to one for the current class and to zero to all others
                       average='binary')
    oneVsAllRecallAvg[posLabel] += recall / CV_FOLDS


    prec = precision_score(labelBinary, predBinary,
                       pos_label=1,  # We all convert to one for the current class and to zero to all others
                       average='binary')
    oneVsAllPrecAvg[posLabel] += prec / CV_FOLDS


    print('Class %d ONE-VS-ALL' % posLabel)
    print('F1       : %g' % f1score)
    print('Recall   : %g' % recall)
    print('Precision: %g' % prec)
    print('------')

  print('================')

print('\n\n')
print('\n\n')
print('Class counts:')
classQtys = getClassCounts(allLabels)
print(classQtys)
print('Class ratios:')
cr = getClassRatios(allLabels)
crk = list(cr.keys())
crk.sort()
for c in crk:
  print(c, formatVal(cr[c]))

print('Multi-class averaging %s' % AVG_TYPE)
print('F1        : %g' % multiClassF1Avg)
print('Recall    : %g' % multiClassRecallAvg)
print('Precision : %g' % multiClassPrecAvg)
print('================')
print('\n\n')

print('Class-specific ONE-VS-ALL scores:')
for posLabel in LABEL_LIST:
  print('Class: ', posLabel)
  print('F1       : %g' % (oneVsAllF1Avg[posLabel]))
  print('recall   : %g' % (oneVsAllRecallAvg[posLabel]))
  print('precision: %g' % (oneVsAllPrecAvg[posLabel]))
  print('================')

if outTableFileName is not None:
  with open(outTableFileName, 'w') as of:
    of.write('Class\tN\tPrec.\tRecall\tF-score\n')
    for posLabel in LABEL_LIST:
      if isMultiClass:
        className = SPAN_ANNOT_TYPE_NAME[posLabel]
      else:
        className = 'Irrelevant' if posLabel == 0 else 'Comment discussion'

      of.write('%s\t%d\t%s\t%s\t%s\n' % (className,
                                         classQtys[posLabel],
                                         formatVal(oneVsAllPrecAvg[posLabel]),
                                         formatVal(oneVsAllRecallAvg[posLabel]),
                                         formatVal(oneVsAllF1Avg[posLabel])))

    of.write('%s\t%d\t%s\t%s\t%s\n' % ('Weighted avg.',
                                       sum(classQtys.values()),
                                       formatVal(multiClassPrecAvg),
                                       formatVal(multiClassRecallAvg),
                                       formatVal(multiClassF1Avg)))

print('Normalized confusion matrix:')
print(confMatrix)

clfAll = createClassifier(modelType)
pipeAll = Pipeline([('vectorizer', vectorizer),
                   ('tfidf', TfidfTransformer()),
                   ('clf', clfAll)])

pipeAll.fit(allEntries, allLabels)
with open(outFileName, 'wb') as f:
  pickle.dump(pipeAll, f)

print(eli5.format_as_text(eli5.explain_weights(clfAll, top=TOP_K_MODEL_FEATURES, vec=vectorizer)))

scores = cross_val_score(pipeAll, allEntries, allLabels, scoring="accuracy", cv=CV_FOLDS)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if ERR_SAMPLE_QTY > 0:
  print('****************************')
  print('SAMPLE MISTAKE:')
  print('****************************')

  labelQty = len(LABEL_LIST)
  for trueClass in range(labelQty):
    for predClass in range(labelQty):
      key = (trueClass, predClass)
      if key in confSampleDict:

        for textOrig, textProc, origPred, predExplan in sampleWithoutRepl(confSampleDict[key], ERR_SAMPLE_QTY):
          print('-----------------------------------------------------')
          print('True class %d Predict class %d' % (trueClass, predClass))
          print('-----------------------------------------------------')
          print(textOrig)
          print('-----------------------------------------------------')
          print(predExplan)
          print('================================================')