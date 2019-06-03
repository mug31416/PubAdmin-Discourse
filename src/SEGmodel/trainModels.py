#!/usr/bin/env python3
import os, sys, gc
import pickle
import argparse
import spacy
import numpy as np

from dataLoader import *
from commonForModeling import *
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
from pystruct.learners import OneSlackSSVM
from pystruct.learners import SubgradientSSVM

from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

SPACY_MODEL_TYPE = "en_core_web_sm"

THREAD_QTY=3
MAX_ITER=100
C=0.1
NOSEQ_MOD = "mlp"

def getCacheFileName(feat_cache_dir, file_name, suff=None):

  return os.path.join(feat_cache_dir, os.path.basename(file_name)) + ('.' + suff if suff is not None else '')


def main(argList):

  parser = argparse.ArgumentParser(description='Training CRF model on GLOSS section data')

  parser.add_argument('--feat_cache_dir', type=str, default=None,
                     metavar = 'directory to cache features', help='directory to cache features')
  parser.add_argument('--train_file',  type=str, required=True,
                      metavar = 'training file', help='An input JSONL file to train model')
  parser.add_argument('--model_file',  type=str, default=None,
                      metavar = 'model file', help='A file to store load/the model')
  parser.add_argument('--test_file',  type=str, required=True,
                      metavar = 'test file', help='An input JSONL file to test model')
  parser.add_argument('--chunk_size', type=int, default=200,
                     metavar = 'cache data chunk size', help='A number of entries in a single cache data chunk')
  parser.add_argument("-s", "--stopwords",   type=str, default=None,
                      metavar = 'stopword file', help='A stopword file.')
  parser.add_argument("--feat_qty", type=int, default=2001,
                      metavar='# of features', help='A number of features')
  parser.add_argument("--head_feat_qty", type=int, default=101,
                      metavar='# of header features', help='A number of features from header')
  parser.add_argument('--use_tfidf', dest='use_tfidf',
                      action='store_true', help='use TFxIDF transform')
  parser.add_argument('--use_simple_feat', dest='use_simple_feat',
                      action='store_true', help='use simple features')
  parser.add_argument('--use_nodseq_model', dest='use_nodseq_model',
                      action='store_true', help='use non-sequence model')
  parser.add_argument('--use_bert', 
                      action='store_true', help='use BERT')
  parser.add_argument('--bert_model_path', type=str, default=None,
                      metavar='BERT model path', help='a path to fine-tuned BERT model')
  parser.add_argument('--use_head_bert', 
                      action='store_true', help='use head BERT')
  parser.add_argument('--no_wordfeat', 
                      action='store_true', help='do not use regular word features')
  parser.add_argument('--label',
                      required=True, type=str,
                      choices=['COMMENT_DISCUSSION', 'CHANGE_IN_RULE', 'NO_CHANGE_IN_RULE', 'ALL'],
                      help='Annotation type to evaluate')

  args = parser.parse_args(argList)

  featCacheDir = args.feat_cache_dir

  if featCacheDir is not None:
    if not os.path.exists(featCacheDir):
      print('Creating cache directory:', featCacheDir)
      os.makedirs(featCacheDir)
    else:
      print('Cache directory', featCacheDir, ' already exists')
  else:
    print('Not using feature cache!')


  text2Feat = TextTransformer(spacy.load(SPACY_MODEL_TYPE, disable=['parser', 'ner']),
                              procTextSimple if args.use_simple_feat else procTextFancy,
                              getStopWordSet(args.stopwords) if args.stopwords is not None else set())

  if featCacheDir is not None:

    testFeatFileCache = getCacheFileName(featCacheDir, args.test_file)
    trainFeatFileCache = getCacheFileName(featCacheDir, args.train_file)
    featurizerFileCache = getCacheFileName(featCacheDir, 'featurizer', 'mdl')

    trainFeatFileCacheExists = chunkCheckDataExists(trainFeatFileCache)
    testFeatFileCacheExists = chunkCheckDataExists(testFeatFileCache)
    featurizerFileCacheExists = os.path.exists(featurizerFileCache)

  else:

    testFeatFileCache = None
    trainFeatFileCache = None
    featurizerFileCache = None

    trainFeatFileCacheExists = False
    testFeatFileCacheExists = False
    featurizerFileCacheExists = False

  modelFileExists = (args.model_file is not None) and (os.path.exists(args.model_file))

  needTestData = not testFeatFileCacheExists
  needTrainingData = not trainFeatFileCacheExists
  needFeaturizer = needTestData or needTestData

  if modelFileExists:

    if needTestData:
      if featurizerFileCacheExists:
        needTrainingData = False

      else:
        needTrainingData = True

  needRawTrainingData = needTrainingData and not trainFeatFileCacheExists

  print('Need featurizer:', needFeaturizer)
  print('Need training data:', needTrainingData)
  print('Need testing data:', needTestData)
  print('Need RAW training data:', needRawTrainingData)

  featurizer = None

  if needFeaturizer and featurizerFileCacheExists:
    with open(featurizerFileCache, 'rb') as f:
      featurizer = pickle.load(f)

  if needTrainingData:

    if needRawTrainingData:

      trainSentData, trainHeaderTextData, trainHeaderFlagData, trainLabs = loadJSONL(args.train_file)

      if featurizer is None:

        featurizer = FeaturizerCRF(args.feat_qty, args.head_feat_qty, args.use_tfidf, 
                                   useWordFeatures = not args.no_wordfeat, 
                                   useBERT = args.use_bert, useHeadBERT = args.use_head_bert,
                                   bertModelPath = args.bert_model_path)

        print('Fitting the featurizer')
        featurizer.fit(text2Feat, trainSentData, trainHeaderTextData)
        if featurizerFileCache is not None:
          with open(featurizerFileCache, 'wb') as f:
            pickle.dump(featurizer, f)

      print('Featurizing training data')
      xTrain, yTrainDict = featurizer.transform(text2Feat, trainSentData, trainHeaderTextData, trainHeaderFlagData), trainLabs
      print('Done!')
    
      if trainFeatFileCache is not None:

        chunkSaveData(trainFeatFileCache, xTrain, trainLabs, args.chunk_size)

  else:

    xTrain, yTrainDict = chunkLoadData(trainFeatFileCache)


  yTrain = yTrainDict[args.label]

  if modelFileExists:

    print('Loading model')

    with open(args.model_file, 'rb') as f:
      model = pickle.load(f)

    print('Done')

  else:

    if args.use_nodseq_model:

      yTrain = np.array(flattenList(yTrain))
      xTrain = np.array(flattenList(xTrain))

    if args.use_nodseq_model:

      if NOSEQ_MOD=="svm":
        print('Using SVM model')
        model = LinearSVC(max_iter=MAX_ITER)

      if NOSEQ_MOD=="mlp":
        print('Using MLP model')
        model = MLPClassifier(max_iter=MAX_ITER, verbose=True)
      if NOSEQ_MOD=="mlp2":
        print('Using 2-layer MLP model')
        model = MLPClassifier(hidden_layer_sizes=(50,50 ),max_iter=MAX_ITER, verbose=True)
      if NOSEQ_MOD=="log":
        print('Using logistic')
        model = MLPClassifier(activation="identity",max_iter=MAX_ITER, verbose=True)

    else:

      print('Using CRF model')

      #model = SubgradientSSVM(model=ChainCRF(), n_jobs=1, C=C, max_iter=MAX_ITER, verbose=1, batch_size=10)
      #model = FrankWolfeSSVM(model=ChainCRF(), n_jobs = 1, C=C, max_iter=MAX_ITER, verbose=1, batch_mode=True)
      model = OneSlackSSVM(model=ChainCRF(), n_jobs = THREAD_QTY, C=C, max_iter=MAX_ITER, verbose=1)

    print('Fitting model')
    model.fit(xTrain, yTrain)
    print('Done')

    if args.model_file is not None:
      modelDir = os.path.dirname(args.model_file)
      if not os.path.exists(modelDir):
        os.makedirs(modelDir)
      with open(args.model_file, 'wb') as f:
        pickle.dump(model, f)


  if not testFeatFileCacheExists:

    testSentData, testHeaderTextData, testHeaderFlagData, testLabs = loadJSONL(args.test_file)
    xTest, yTestDict = featurizer.transform(text2Feat, testSentData, testHeaderTextData, testHeaderFlagData), testLabs

    if testFeatFileCache is not None:
      chunkSaveData(testFeatFileCache, xTest, yTestDict, args.chunk_size)

  else:
    xTest, yTestDict = chunkLoadData(testFeatFileCache)

  if args.use_nodseq_model:

    xTest = np.array(flattenList(xTest))

    if NOSEQ_MOD=="svm":
      yPredFlat = np.array(model.decision_function(xTest))

    if NOSEQ_MOD in ["mlp","log","mlp2"]:
      yPredFlat = np.array(model.predict(xTest))

  else:

    yPredFlat = np.array(flattenList(model.predict(xTest)))

  yTestFlat, posMap = flattenListExtended(yTestDict[args.label])
  yTestFlat = np.array(yTestFlat)

  #print(posMap)

  if args.use_nodseq_model:

    thres = getThres(yTestFlat, yPredFlat)
    pref = "preds_"+NOSEQ_MOD

    print("AUC:", roc_auc_score(yTestFlat, yPredFlat),
          "Accuracy:", accuracy_score(yTestFlat, yPredFlat > thres ),
          "f1:", f1_score(yTestFlat, yPredFlat > thres),
          "prec:", precision_score(yTestFlat, yPredFlat > thres),
          "recall:", recall_score(yTestFlat, yPredFlat> thres)
          )
  else:

    thres = 0
    pref= "preds_crf"

    print("AUC:", " n.a.",
          "Accuracy:", accuracy_score(yTestFlat, yPredFlat  ),
          "f1:", f1_score(yTestFlat, yPredFlat ),
          "prec:", precision_score(yTestFlat, yPredFlat ),
          "recall:", recall_score(yTestFlat, yPredFlat)
          )

  print("# of 1 examples:", np.sum(yTestFlat == 1))
  print("# of 0 examples:", np.sum(yTestFlat == 0))


  err_dict = {'tp':[],'fp':[],'tn':[],'fn':[]}
  for i in range(len(yPredFlat)):
    if yPredFlat[i] > thres and yTestFlat[i]==1:
      err_dict['tp'].append(i)
    if yPredFlat[i] < thres and yTestFlat[i]==0:
      err_dict['tn'].append(i)
    if yPredFlat[i] > thres and yTestFlat[i]==0:
      err_dict['fp'].append(i)
    if yPredFlat[i] < thres and yTestFlat[i]==1:
      err_dict['fn'].append(i)

  if len(err_dict['fp'])>5:
    randIdx = np.random.choice(err_dict['fp'],5,replace=False)
    for idx in randIdx:
      doc, sec = posMap[idx]
      print('Test FP idx', doc, sec)
  else:
    print('.')
    print('.')
    print('.')
    print('.')
    print('.')

  if len(err_dict['fn'])>5:
    randIdx = np.random.choice(err_dict['fn'], 5, replace=False)
    for idx in randIdx:
      doc, sec = posMap[idx]
      print('Test FN idx', doc, sec)
  else:
    print('.')
    print('.')
    print('.')
    print('.')
    print('.')

  # Save predictions
  with open(os.path.join(os.path.dirname(args.model_file),pref+"_"+args.label+".pkl"), 'wb') as f:
    pickle.dump((yTestFlat,yPredFlat,thres), f)


if __name__ == '__main__':
  main(sys.argv[1:])

#Test FP idx [ 6058  5249 59341 28590 14311]
#Test FN idx [55691 18266  3909  4096   428]
#sed -n '6,6p' ../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl
