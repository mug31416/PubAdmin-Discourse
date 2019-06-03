import os
from utils.io import readJson
from utils.api import *
from collections import namedtuple

DEFAULT_START_YEAR=2003

NUM_OCCUR = '# of occurrences: '

CHUNK_ID_FIELD = "chunkId"
START_OFFSET_FIELD = "startOffset"
END_OFFSET_FIELD = "endOffset"

SECTION_ID_FIELD = 'sectId'
SECT_NUM_FIELD = 'sectNum'
TITLE_FIELD = 'title'
LEVEL_FIELD = 'level'
TITLE_1ST_LEVEL_FIELD = 'title1stLev'
TITLE_2D_LEVEL_FIELD = 'title2dLev'
SECTION_CHUNK = 'sectChunk'
DOCKET_ID_FIELD = DOCKET_ID_KEY

LUCENE_SCORE_FIELD = "luceneScore"
NORM_LUCENE_SCORE_FIELD = "normLuceneScore"
SCORE_FIELD = "score"
TITLE_SECT_FLAG = "titleSectFlag"

DOC_META_SHORT_FILT_BY_YEAR_PATTTERN = 'documentsMetaShort%dFinal=%d.json'
DOC_META_SHORT_PATTERN = 'documentsMetaShortFinal=%d.json'
DOC_META_FULL_PATTERN = 'documentsMetaFullFinal=%d.json'

BINARY_TYPE = 'binaryType'
TYPE = 'type'

def replNewLines(s, repStr=' '):
  s = s.replace('\n', repStr)
  s = s.replace('\r', repStr)
  return s

def getDirName(docketId):
  """Converts docket ID to a directory name.

    :param docketId:   a docket ID
    :return a directory: name
  """
  return docketId.replace('-', os.sep)

def getDocPath(docketDir, docketId, docId, isFinal, ext):
  """Constructs a full path and a full file name to store the document.

  :param docketDir: docket root directory
  :param docketId: docket ID
  :param docId: document ID
  :param isFinal: True for final rules and False for proposals
  :param ext: extension, i.e., pdf or html

  :return a tuple: directory name, full path to the file
  """
  subDir = '' if isFinal else 'proposals'
  docDir = os.path.join(docketDir, ext, getDirName(docketId), subDir)
  docFile = os.path.join(docDir, docId + '.' + ext)
  return docDir, docFile

def getDocInfoFromMetaFull(docketDir, startYear):
  """Create an iterator to retrieve document information from full-meta JSONs.

  :param docketDir: docket root directory.
  :param startYear: a starting year, if it's <=0, the year isn't used for filtering
                    A catch is that some dates are empty: If you specify a small date > 0,
                    they would still be ignored!!!
  :return: This function yields a tuple:
          docket ID, document ID, posted date, final flag, file type: html, pdf or None, fileName or None
  """
  for isFinal in [False, True]:
    inpFile = os.path.join(docketDir, DOC_META_FULL_PATTERN % int(isFinal))

    docs = readJson(inpFile)

    for e in docs[DOCUMENTS_KEY]:
      if not checkPostedYear(e, startYear):
        continue
      docId = e[DOCUMENT_ID_KEY][VALUE_KEY]
      docketId = e[DOCKET_ID_KEY][VALUE_KEY]
      #if POSTED_DATE_KEY not in e:
      #  print('Missing %s key in docId %s file %s' % (POSTED_DATE_KEY, docId, inpFile))
      postedDate = e[POSTED_DATE_KEY] if POSTED_DATE_KEY in e else None

      if hasFormat(e, REQUEST_HTML):
        ext = REQUEST_HTML
      elif hasFormat(e, REQUEST_PDF):
        ext = REQUEST_PDF
      else:
        ext = None
      fileName = None if ext is None else getDocPath(docketDir, docketId, docId, isFinal, ext)[1]

      yield docketId, docId, postedDate, isFinal, ext, fileName

def getDocInfoFromMetaShort(docketDir, startYear):
  """Create an iterator to retrieve document information from short-meta JSONs.

  :param docketDir: docket root directory.
  :param startYear: a starting year, if it's <=0, the year isn't used for filtering
                    A catch is that some dates are empty: If you specify a small date > 0,
                    they would still be ignored!!!
  :return: This function yields tuples: docket ID, document ID, postedYear, final flag
  """
  for isFinal in [False, True]:
    inpFile = os.path.join(docketDir, DOC_META_SHORT_PATTERN % int(isFinal))

    docs = readJson(inpFile)

    for e in docs[DOCUMENTS_KEY]:
      docId = e[DOCUMENT_ID_KEY]
      docketId = e[DOCKET_ID_KEY]
      postedDate = e[POSTED_DATE_KEY]

      if not checkPostedYear(e, startYear):
        continue

      yield docketId, docId, postedDate, isFinal

def checkPostedYear(doc, startYear):
  if startYear <= 0:
    return True
  yr = getPostedYearFromJSON(doc)

  if yr is None or yr == '':
    return False

  return yr >= startYear

def dictToClass(className, d):
  """Convert dictionary to class.

  :param className a class name
  :param d  a dictionary

  :return an object whose attributes are dictionary keys and attribute
          values are values of these keys in the input dictionary.

  """
  return namedtuple(className, d.keys())(*d.values())

from random import shuffle

def sampleWithoutRepl(iterObj, sampleQty):
  x = [e for e in iterObj]
  shuffle(x)
  return [x[i] for i in range(min(sampleQty, len(x)))]

def seedAll(val):
  from random import seed
  import numpy.random as rnd

  seed(val)
  rnd.seed(val)


def getDocketIdFromDocId(docId):
  return "-".join(docId.split("-")[0:-1])



