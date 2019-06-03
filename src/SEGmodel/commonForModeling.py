import sys, gc
sys.path.append('.')
sys.path.append('..')

from random import randint

from utils.misc import *
from utils.spacyAnnot import *
from utils.parse import *
from dataLoader import LABEL_DICT


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.modeling import BertForPreTraining

import torch
import tqdm
import numpy as np

from fullTextIndex.textProcForIndex import *


AVG_TYPE = 'weighted'
PRINT_FORMAT = '%.2g'
MAX_BERT_LEN=128
BERT_BATCH_QTY=16
CLS_TOK='[CLS]'
SEP_TOK='[SEP]'

def formatVal(v):
  return str(PRINT_FORMAT % v)

def getDocPassTextForModel(tokExtr,
                           useTitleSectFlag, titleSectFlag,
                           useTitle, title,
                           useText, text):
  res = []

  if useTitle:
    for tok in tokExtr.getTokens(title):
      res.append(tok)

  if useText:
    for tok in tokExtr.getTokens(text):
      res.append(tok)

  if useTitleSectFlag:
    res.append('titleSectFlag_' + str(titleSectFlag))

  return ' '.join(res)


def getClassCounts(labels):
  """Labels are assumed to be a numpy array"""
  labelSet = set(labels)
  res = dict()

  for lab in labelSet:
    res[lab] = sum(labels == lab)

  return res


def getClassRatios(labels):
  """Labels are assumed to be a numpy array"""
  classQtys = getClassCounts(labels)
  totQty = sum(classQtys.values())
  return {k : v/totQty for k, v in classQtys.items()}


def getNegSamples(tokExtr, docketDir, startYear,
                  sectTitleTrainSet, docketIdSet,
                  negSampleQty,
                  sentQty, sentSkip):
  sampleRes = []

  if negSampleQty > 0:
    print('Sampling negative examples')

    chunkNum = 0
    docNum = 0

    for docketId, docId, postedDate, isFinal, ext, htmlFileName in getDocInfoFromMetaFull(docketDir,
                                                                                          startYear):
      if docketId not in docketIdSet:
        continue
      if ext != 'html' or isFinal:
        continue
      print('Processing: ', htmlFileName)
      if not os.path.exists(htmlFileName):
        print('Ignoring missing file!!!!')
        continue

      docNum += 1
      sectNums, sectNames, sectParas = partialParse(htmlFileName)

      for i in range(len(sectNames)):
        # sectParas is a list of list of paragraphs
        fullText = '\n'.join(sectParas[i])
        sectTitle = sectNames[i]
        for qStartOff, qEndOff, docChunkDispTxt, docChunkQueryTxt \
          in splitTextIterator(tokExtr, fullText, sentQty, sentSkip):
          chunkNum += 1

          e = {BINARY_TYPE: 0,
               TYPE: SPAN_TYPE_SKIP,
               TITLE_SECT_FLAG: specialSectTitleFlag([sectTitle], sectTitleTrainSet),
               TITLE_FIELD: sectNames[i],
               SPACY_TEXT_FIELD: docChunkDispTxt}
          # Reservoir sampling
          if len(sampleRes) < negSampleQty:
            sampleRes.append(e)
          else:
            replPos = randint(0, chunkNum - 1)
            if replPos < negSampleQty:
              sampleRes[replPos] = e

    print('# of docs: %d # of chunks: %d # of chunks per doc: %g' % (docNum, chunkNum, float(chunkNum)/docNum))

  return sampleRes


def isListOrNumpyArray(inp):
    return isinstance(inp, list) or isinstance(inp, np.ndarray) or isinstance(inp, tuple)

def numpyToListRecursive(inp):
    if isListOrNumpyArray(inp):
        return [numpyToListRecursive(e) for e in inp]
    else:
        return int(inp) if isinstance(inp, np.int64) else inp
    
def listToNumpyRecursive(inp):
    if isListOrNumpyArray(inp):
        return np.array([listToNumpyRecursive(e) for e in inp])
    else:
        return inp

def chunkSaveData(filePrefixName, x, y, chunkSize):
  qty = len(x)

  #print(y)

  for l in LABEL_DICT:
    if qty != len(y[l]):
      raise Exception('x and y are of different lengths for %s: %d %d' % (l, len(x), len(y[l])))

  chunkQty = int((qty + chunkSize - 1) / chunkSize)

  np.save(filePrefixName + '.qty', chunkQty)
    
  for cid in range(chunkQty):
    s = cid * chunkSize
    e = min(s + chunkSize, qty)
    np.save(filePrefixName + "_%d_x.npy" % cid, x[s:e])
    for l in LABEL_DICT:
      np.save(filePrefixName + "_%d_%s_y.npy" % (cid,l), y[l][s:e])

def chunkCheckDataExists(filePrefixName):

  fn = filePrefixName + '.qty.npy'

  print('Checking cache for:', filePrefixName)
  if os.path.exists(fn):
    chunkQty = int(np.load(fn))
    print('Qty file %s exists, # of chunks %d' % (fn, chunkQty))

    for cid in range(chunkQty):
      fns = [filePrefixName + "_%d_x.npy" % cid]
      for l in LABEL_DICT:
        fns.append(filePrefixName + "_%d_%s_y.npy" % (cid,l))

      for fn in fns:
        exfl = os.path.exists(fn) 
        print('File %s exists flag %d' % (fn, exfl))
        if not exfl: 
          return False
  else:
    return False

  return True

def chunkLoadData(filePrefixName):

  chunkQty = int(np.load(filePrefixName + '.qty.npy'))

  x = []

  yDict = {}
  for l in LABEL_DICT:
    yDict[l]=[]

  for cid in range(chunkQty):
    print(filePrefixName, 'loading chunk:', cid)
    x.extend(list(np.load(filePrefixName + "_%d_x.npy" % cid)))

    for l in LABEL_DICT:
      yDict[l].extend(list(np.load(filePrefixName + "_%d_%s_y.npy" % (cid,l))))

  for l in LABEL_DICT:
    yDict[l] = np.array(yDict[l])

  return np.array(x), yDict

# Processes input text and generates
# a list of features (unigrams and POS-enriched bigrams)
def procTextSimple(nlp, stopList, text):
  text = text.replace("’", "'");
  parsedSent = nlp(text)

  unigr = []
  for token in parsedSent:
    if token.pos_ != 'PUNCT' and \
        token.lemma_.strip() != '':  # Sometimes we get these annoying
      unigr.append(token.lemma_.lower())

  res = [tok for tok in unigr if tok not in stopList]

  for i in range(1, len(unigr)):
    tok1 = unigr[i-1]
    tok2 = unigr[i]

    res.append(tok1 + '_' + tok2)

  #print(res)

  return ' '.join(res)

# Processes input text and generates
# a list of features (unigrams and POS-enriched bigrams)
def procTextFancy(nlp, stopList, text):
  text = text.replace("’", "'");
  parsedSent = nlp(text)

  unigr = []
  for token in parsedSent:
    if token.pos_ != 'PUNCT' and \
       token.lemma_.strip() != '': # Sometimes we get these annoying
       # virtually empty tokens which we need to ignore
      unigr.append(token)

  res = []

  for token in unigr:
    if not token.lower_ in stopList:
      res.append(token.pos_ + '_' + token.lemma_)

  # Typically we don't need bigrams containing stop words
  # However, bigrams with adverbial particles and prepositions
  # can be useful so we keep them
  for i in range(1, len(unigr)):
    tok1 = unigr[i-1]
    tok2 = unigr[i]
    # The only stop words that we consider here are averbial particles
    # or prepositions (Scipy somehow mark all prepositions using ADP)
    isStop1 = tok1.lower_ in stopList
    isStop2 = tok2.lower_ in stopList

    acceptStop1 = (not isStop1) or tok1.pos_ == 'ADP'
    acceptStop2 = (not isStop2) or tok2.pos_ == 'ADP'

    if acceptStop1 and acceptStop2 and (not (isStop1 and isStop2)) :
      res.append(tok1.pos_ + '_' + tok1.lemma_ + '_' + tok2.pos_ + '_' + tok2.lemma_)
      res.append(tok1.pos_ + '_' + tok2.pos_)

  #print(res)

  return ' '.join(res)


# Every step in a pipeline needs to be a "transformer".
# Define a custom transformer to
# 1) clean up text using spaCy
# 2) extract unigrams
# 3) extract bigrams
class TextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def __init__(self, nlp, procTextFunc, stopList):
      self._nlp = nlp
      self._stopList = stopList
      self._procText = procTextFunc

    def transform(self, X, printProgress=False):
        res = []

        for text in tqdm.tqdm(X) if printProgress else X:
            res.append(self._procText(self._nlp, self._stopList, text))
        return res

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def flattenList(listOfLists):
  return [e for subList in listOfLists for e in subList]

def flattenListExtended(listOfLists):
  elems = []
  posTuple = []
  sec = 0
  for sublist in listOfLists:
    idx = 0
    for e in sublist:
      elems.append(e)
      posTuple.append((sec,idx))
      idx = idx +1
    sec = sec + 1

  return elems, posTuple


def getModelNamesByPrefix(modelPrefix):
  return (modelPrefix + '.ml', modelPrefix + '.vect', modelPrefix + '.tfidf')


class FeaturizerCRF:

  def __init__(self, featQty, headFeatQty, useTfIdfTransform=False, useWordFeatures=True,
               useBERT=False, useHeadBERT=False,
               bertModelPath=None,
               torch_device='cuda',
               bertModelType='bert-base-uncased'):


    self.useWordFeatures = useWordFeatures
    if self.useWordFeatures:
      self.featQty = featQty
      self.countVect = CountVectorizer(ngram_range=(1, 1))
      self.tfidf = TfidfTransformer() if useTfIdfTransform else None

      self.headFeatQty = headFeatQty
      self.headCountVect = CountVectorizer(ngram_range=(1, 1))
      self.headTfidf = TfidfTransformer() if useTfIdfTransform else None

    self.useBERT = useBERT
    self.useHeadBERT = useHeadBERT
    if useBERT or useHeadBERT:
      self.torch_device=torch.device(torch_device)
      if bertModelPath is not None:
        print('Loading fine-tuned model from file:', bertModelPath)
        self.bertModelWrapper = BertForPreTraining.from_pretrained(bertModelType)
        self.bertModelWrapper.load_state_dict(torch.load(bertModelPath))
      else:
        print('Loading standard pre-trained model')
        self.bertModelWrapper = BertForMaskedLM.from_pretrained(bertModelType)

      self.bertModelWrapper.eval()
      self.bertModelWrapper.to(torch_device)
      self.bert_tokenizer = BertTokenizer.from_pretrained(bertModelType, do_lower_case=True)


  def batchForBert(self, sent_list, max_len=MAX_BERT_LEN):
    batch_qty = len(sent_list)
    batch_data = []
    batch_max_seq_qty = 0
    for text in sent_list:
      bert_toks = self.bert_tokenizer.tokenize(text)
      # The text may or may not contain the CLS_TOK and SEP_TOK
      if len(bert_toks) == 0 or bert_toks[0] != CLS_TOK:
        bert_toks.insert(0, CLS_TOK)

      bert_toks = bert_toks[0:MAX_BERT_LEN]

      if bert_toks[-1] != SEP_TOK: 
        if len(bert_toks) < MAX_BERT_LEN:
          bert_toks.append(SEP_TOK)
        else:
          bert_toks[-1] = SEP_TOK

      tok_ids = self.bert_tokenizer.convert_tokens_to_ids(bert_toks)
      batch_max_seq_qty = max(batch_max_seq_qty, len(tok_ids))
      batch_data.append(tok_ids)

    tok_ids_batch = np.zeros( (batch_qty, batch_max_seq_qty), dtype=np.int64) # zero is a padding symbol\n",
    tok_mask_batch = np.zeros( (batch_qty, batch_max_seq_qty), dtype=np.int64)

    for k in range(batch_qty):
      tok_ids = batch_data[k]
      tok_qty = len(tok_ids)
      tok_ids_batch[k, 0:tok_qty] = tok_ids
      tok_mask_batch[k, 0:tok_qty] = np.ones(tok_qty)

    tok_ids_batch = torch.from_numpy(tok_ids_batch).to(device=self.torch_device)
    tok_mask_batch = torch.from_numpy(tok_mask_batch).to(device=self.torch_device)
  
    return tok_ids_batch, tok_mask_batch

  @staticmethod
  def toDenseWithHashing(sparseVect, denseDim):
    shape = sparseVect.shape
    if (len(shape) != 2 or shape[0] != 1):
      raise Exception('Invalid shape of the sparse vect %s, expected a one-row matrix' % (str(shape)))
    res = np.zeros(denseDim)
    for c in sparseVect.nonzero()[1]:
      ti = hash(c) % denseDim
      res[ti] = res[ti] + sparseVect[0, c]
    return res


  def _fit(self, text2Feat, listOfSentListsTrain, countMod, tfidfMod):

    # Fit the transformers
    sentAll = flattenList(listOfSentListsTrain)

    print('Fitting data for: %d sentences' % len(sentAll))

    sentAllProc = text2Feat.transform(sentAll, printProgress=True)
    countMod.fit(sentAllProc)

    if tfidfMod is not None:
      sentAllCount = countMod.transform(sentAllProc)
      tfidfMod.fit(sentAllCount)


  def _transform(self, text2Feat, listOfSentLists, fQty, countMod, tfidfMod):

    res = []
    lstQty = 0

    print('Transforming data for: %d sentence lists' % len(listOfSentLists))
    for oneSentLst in tqdm.tqdm(listOfSentLists):

      val = text2Feat.transform(oneSentLst)
      val = countMod.transform(val)

      if tfidfMod is not None:
        val = tfidfMod.transform(val)

      dval = np.array([FeaturizerCRF.toDenseWithHashing(v, fQty) for v in val])


      res.append(dval)
      lstQty = lstQty + 1
      # print('%d sentence lists are featurized' % lstQty)

    return np.array(res)


  def fit(self, text2Feat, listOfSentListsTrain, listOfHeaderTextLists):

    if self.useWordFeatures:

      # Fit the transformers to sentences
      self._fit(text2Feat, listOfSentListsTrain, self.countVect, self.tfidf)

      # Fit transformers to headers
      self._fit(text2Feat, listOfHeaderTextLists, self.headCountVect, self.headTfidf)


  def getBertStates(self, listOfSentLists):
    res = []

    print('BERT-transforming data for: %d sentence lists' % len(listOfSentLists))
    flatSentList = []
    listQtys = []
    for oneSentLst in tqdm.tqdm(listOfSentLists):
      listQtys.append(len(oneSentLst))
      flatSentList.extend(oneSentLst)

    numOfBatch = int( (len(flatSentList) + BERT_BATCH_QTY - 1) / BERT_BATCH_QTY )

    bert = self.bertModelWrapper.bert

    tmpOut = []

    for bid in tqdm.tqdm(range(numOfBatch)):
      bs = bid * BERT_BATCH_QTY
      be = min(bs + BERT_BATCH_QTY, len(flatSentList))

      tok_ids_batch, tok_mask_batch = self.batchForBert(flatSentList[bs:be])
      seg_ids = torch.zeros_like(tok_ids_batch, device=self.torch_device)

      # Transformations from the main BERT model
      #print(tok_ids_batch.size(), tok_mask_batch.size(), seg_ids.size(), '@@@')
      _, pooled_output = bert(tok_ids_batch, seg_ids, attention_mask=tok_mask_batch, output_all_encoded_layers=False)

      tok_ids_batch.detach()
      tok_mask_batch.detach()
      seg_ids.detach()

      tmpOut.append(pooled_output.detach().cpu().numpy())

      torch.cuda.empty_cache()
      gc.collect()

    flatBertPred = np.vstack(tmpOut)

    print(flatBertPred.shape, len(listQtys))

    start = 0
    for qty in listQtys:
      res.append(flatBertPred[start:start+qty]) 
      start += qty

    return res
      


  def transform(self, text2Feat, listOfSentLists, listOfHeaderTextLists, listOfHeaderFlagLists):
    # BERT-transform sentences
    if self.useBERT:
      bertStates = self.getBertStates(listOfSentLists)
    if self.useHeadBERT:
      bertHeadStates = self.getBertStates(listOfHeaderTextLists)

    if self.useWordFeatures:
      # Transform sentences
      listOfTransSentLists = self._transform(text2Feat, listOfSentLists,  self.featQty,
                                            self.countVect, self.tfidf)

      # Transform headers
      listOfTransHeaderTextLists = self._transform(text2Feat, listOfHeaderTextLists,  self.headFeatQty,
                                                  self.headCountVect, self.headTfidf)

      # Transform header flags
      listOfTransHeaderFlagLists=[]
      for lst in listOfHeaderFlagLists:
        listOfTransHeaderFlagLists.append(np.array(lst).reshape(-1,1))

    # Compose the final representation
    res = []
    numSentLists = len(listOfSentLists)

    print('Composing data for: %d sentence lists' % numSentLists)

    for lst in tqdm.tqdm(range(numSentLists)):
      val = []
      if self.useWordFeatures:
        val = np.hstack((listOfTransSentLists[lst],
                       listOfTransHeaderTextLists[lst],
                       listOfTransHeaderFlagLists[lst]))
      if self.useBERT:
        if len(val):
          val = np.hstack((val, bertStates[lst]))
        else:
          val = bertStates[lst]
      if self.useHeadBERT:
        if len(val):
          val = np.hstack((val, bertHeadStates[lst]))
        else:
          val = bertHeadStates[lst]
      res.append(val)

    return np.array(res)


def getThres(labels,probs):

  best_score = -1
  best_thres = -1

  for x in np.linspace(0, 1.0, num=999):
      scr = f1_score(labels, probs >= x)
      if scr > best_score:
          best_thres = x
          best_score = scr

  return best_thres
