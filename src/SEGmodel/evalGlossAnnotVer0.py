#!/usr/bin/env python3
import sys, argparse, os, pickle
sys.path.append('.')

import numpy as np
import pickle

from fullTextIndex.textProcForIndex import *
from utils.spacyAnnot import *
from utils.misc import *

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


EXAMPLE_DOCUMENTS = ["EPA-HQ-OAR-2012-0133-0067.txt","EPA-R04-OAR-2017-0436-0012.txt","EPA-HQ-SFUND-1990-0011-0136.txt"]
TEST_DOCUMENTS = ["EPA-HQ-OLEM-2016-0177-0089.txt","EPA-R09-OAR-2015-0622-0063.txt",
                  "EPA-HQ-TRI-2011-0174-0011.txt","EPA-HQ-OAR-2009-0299-0070.txt"]
EXPERT_OWNER = "5c491ecbee951f365779e552"
EXCLUDED_OWNERS = ["5ad5082accf5e87c942303c4","58a0bf4f8424bd4f65e2be57","5c5c769cee951f365779ead2"]
ANNO_TYPES = ['COMMENT_DISCUSSION','CHANGE_IN_RULE','NO_CHANGE_IN_RULE']

tokExtr=TokenExtractor(STOP_WORDS)

JSON_MARKER = '###JSON###'
META_MARKER = '##META##'

SENT_QTY = 4
MIN_ANNOT_OVERLAP = 0.8

ERROR_FILE_NAME_PREF='error_annot'

class LineIter:

  def eof(self):
    return self.cur >= len(text)

  def getLine(self):

    return self.text[self.cur:self.next]

  def __init__(self, text, sep = '\n'):

    self.text = text
    self.sep = sep
    self.cur = 0
    self._nextSepOffset()

  def getStartOffset(self):

    return self.cur

  def goNext(self):

    if not self.eof():
      self.cur = self.next + 1
      self._nextSepOffset()


  def _nextSepOffset(self):

    if self.cur < len(self.text):

      self.next = self.text.find(self.sep, self.cur)
      if self.next == -1:
        self.next = len(text)


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

glossJSON = readJson(args.glossJSON)
docSetJSON = readJson(args.docSetJSON)
sectTitleTrainSet = loadTrainTitles(args.sectTitleTrainJSONL)

docSet = docSetJSON["documents"].keys()
print(docSet)

modelFileName = args.modelFile
model = pickle.load(open(modelFileName, 'rb'))
print('Loaded model from file: ', modelFileName)

avgType = args.avgType

error_file_n = 0

allRelev = []
allPred = []

for doc_id, doc_json in glossJSON['documents'].items():

  doc_name = doc_json["name"]
  docketId = getDocketIdFromDocId(doc_name)
  print(docketId)

  if docketId not in docSet:
    print(f'Ignoring docket {docketId} b/c it is not in the set')
    continue

  print(doc_json.keys())
  text = doc_json['plainText']
  # with open('test.txt', 'w') as f:
  #   f.write(text.replace('\n','@'))

  annots = doc_json['annotations']
  # Sample annotation:
  # '5c82c1e080930b282483df40': {'type': 'COMMENT_DISCUSSION', 'span': [10724, 12016], 'owner': '5c5c7670ee951f365779ead0'},

  textLineIter = LineIter(text, '\n')

  while not textLineIter.eof():

    line = textLineIter.getLine()

    if not line.startswith(JSON_MARKER):
      textLineIter.goNext()
    else:
      setJSON = json.loads(line[len(JSON_MARKER):])
      print(setJSON)

      sectType = setJSON[LEVEL_FIELD]
      prev1stLevelTitle = setJSON[TITLE_1ST_LEVEL_FIELD]
      prev2dLevelTitle = setJSON[TITLE_2D_LEVEL_FIELD]

      titleSectFlag = specialSectTitleFlag([prev1stLevelTitle, prev2dLevelTitle], sectTitleTrainSet)
      sectCompTitle = getCompositeTitle(sectType, prev1stLevelTitle, prev2dLevelTitle, setJSON[TITLE_FIELD])

      textLineIter.goNext()
      if textLineIter.eof():
        break

      startOffset = textLineIter.getStartOffset()

      while not textLineIter.eof() and not textLineIter.getLine().startswith(META_MARKER):
        textLineIter.goNext()

      endOffset = textLineIter.getStartOffset()

      # Here do something with text from start to end

      sectAnnot = []

      for _, e in annots.items():

        owner = e['owner']

        if owner in EXCLUDED_OWNERS:
          print("@@@@@Excluded owner ", owner)
          continue

        if doc_name in TEST_DOCUMENTS and owner != EXPERT_OWNER:
          print("@@@@@Test document, defer annotation to expert")
          continue

        if doc_name in EXAMPLE_DOCUMENTS and owner != EXPERT_OWNER:
          print("@@@@Example document, defer annotation to expert")
          continue

        if e['type'] in ANNO_TYPES:
          annotStart, annotEnd = e['span']

          if annotStart >= startOffset and annotEnd <= endOffset:
            sectAnnot.append( (annotStart, annotEnd) )

          else:
            if annotStart < endOffset and annotEnd > startOffset:
              print('annot:', annotStart, annotEnd, ' vs. sect offsets:', startOffset, endOffset)
              efn = ERROR_FILE_NAME_PREF + '.' + str(error_file_n)
              error_file_n += 1

              with open(efn, 'w') as f:
                f.write(text[annotStart:annotEnd])
              print('Bug: annotation crosses section boundaries for annotator %s, check content in %s, truncating annotation!' % 
                      (owner, efn) )
              annotStart = max(annotStart, startOffset)
              annotEnd = min(annotEnd, endOffset)
              sectAnnot.append((annotStart, annotEnd))

      sectText = text[startOffset:endOffset]

      sectSents = []

      for sentStart, sentEnd, _ in tokExtr.getTokenizedSentences(sectText):
        sectSents.append( (sentStart, sentEnd, sectText[sentStart:sentEnd]))
        #print(sectSents[-1])

      sectSentQty = len(sectSents)
      sentRelev = np.zeros(sectSentQty)
      sentPreds = np.zeros(sectSentQty)

      feats = []
      sentRanges = []

      for sidStart in range(0, sectSentQty, SENT_QTY):
        sidEnd = min(sidStart + SENT_QTY, sectSentQty)
        passText = ' '.join([e[2] for e in sectSents[sidStart:sidEnd]])
        sentRanges.append( (sidStart, sidEnd))

        featTxt = getDocPassTextForModel(tokExtr,
                                         True, titleSectFlag,
                                         True, sectCompTitle,
                                         True, passText)
        feats.append(featTxt)

      preds = model.predict(feats)

      for predId in range(len(preds)):
        sidStart, sidEnd = sentRanges[predId]
        for sid in range(sidStart, sidEnd):
          sentPreds[sid] = preds[predId]


      for ast, aend in sectAnnot:
        origAnnotText = text[ast:aend]
        ast -= startOffset
        aend -= startOffset

        for sid in range(sectSentQty):
          start, end, _ = sectSents[sid]
          s1 = max(start, ast)
          s2 = min(end, aend)
          if s1 < s2 and end > start:
            cov = float(s2 - s1) / (end - start)
            if cov >= MIN_ANNOT_OVERLAP:
              sentRelev[sid] += 1

        sectAnnotText = sectText[ast:aend]
        if (origAnnotText != sectAnnotText):
          print(origAnnotText)
          print('------------')
          print(sectAnnotText)
          print('------------')
          raise Exception('Bug: annotation offset mismatch!')

      print('Ground-truth sentence relevance:')
      print(sentRelev)
      print('Predicted sentence relevance')
      print(sentPreds)

      allRelev.extend(list( (sentRelev > 0).astype(int)))
      allPred.extend(list(sentPreds))

      print('Relevant sentences:')
      for sid in range(sectSentQty):
        if sentRelev[sid] > 0:
          print(sentRelev[sid], sectSents[sid][2])


totQty = len(allRelev)
for i in range(totQty):
  print(str(allRelev[i]) + " " + str(allPred[i]))

f1 = f1_score(allRelev, allPred, average=avgType)
recall = recall_score(allRelev, allPred, average=avgType)
prec = precision_score(allRelev, allPred, average=avgType)

print('F1        : %s' % formatVal(f1))
print('Recall    : %s' % formatVal(recall))
print('Precision : %s' % formatVal(prec))
