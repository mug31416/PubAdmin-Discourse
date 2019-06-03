from collections import namedtuple
import json

from utils.misc import CHUNK_ID_FIELD, START_OFFSET_FIELD, END_OFFSET_FIELD

SPACY_META_FIELD = 'meta'
SPACY_TEXT_FIELD = 'text'
SPACY_SPAN_FIELD = 'spans'
SPACY_LABEL_FIELD = 'label'

MODEL_PRED = 'modelPred'

MASTER_JSON_FILE_NAME_FIELD = 'fileName'
MASTER_JSON_TYPE_FIELD = 'type'
MASTER_JSON_TYPE_SPAN = 'span'
MASTER_JSON_TYPE_YESNO = 'yesno'
MASTER_JSON_IS_SEED_FIELD = 'isSeed'

ANSWER_FIELD = 'answer'
ACCEPT_FLAG = 'accept'
ACCEPT_FIELD = 'accept'

SPAN_TYPE_SKIP = 'skip'
SPAN_TYPE_NEUTRAL = 'neutral'
SPAN_TYPE_ACCEPT = 'accept'
SPAN_TYPE_REJECT = 'reject'

SPAN_ANNOT_TYPES = [SPAN_TYPE_NEUTRAL, SPAN_TYPE_ACCEPT, SPAN_TYPE_REJECT, SPAN_TYPE_SKIP]
SPAN_ANNOT_TYPE_CODE = {
  SPAN_TYPE_SKIP: 0,
  SPAN_TYPE_NEUTRAL: 1,
  SPAN_TYPE_ACCEPT: 2,
  SPAN_TYPE_REJECT: 3
}
SPAN_ANNOT_TYPE_NAME = {
  0: 'Irrelevant',
  1: 'Neutral',
  2: 'Accept',
  3: 'Reject'
}
SPAN_ANNOT_TYPES_IS_COMMENT = [True, True, True, False]

def getCommentAnnotTypes(isComment):
  return [SPAN_ANNOT_TYPES[i]
          for i in range(len(SPAN_ANNOT_TYPES)) if isComment==SPAN_ANNOT_TYPES_IS_COMMENT[i]]

def checkSpacyAccept(docJson):
  return ANSWER_FIELD in docJson and docJson[ANSWER_FIELD] == ACCEPT_FLAG

def checkSpacyWholeDocAnnot(docJson, optText=['yes']):
    if not checkSpacyAccept(docJson ):
      return False
    answArr = docJson[ACCEPT_FIELD]
    if len(answArr) != 1:
      raise Exception('Unexpected number of anwer options in "accept" field (expect only one): '
                      + json.dumps(docJson))
    return answArr[0] in optText

def loadTrainTitles(trainJSONL):

  trainTitles = set()

  for line in open(trainJSONL):
    #print(line)
    oneAnnot = json.loads(line)

    if checkSpacyWholeDocAnnot(oneAnnot):
      trainTitles.add(oneAnnot["text"].lower())

  return trainTitles

from utils.misc import *
from json import dumps

def writeSectTextForAnnot(outFile,
                          docketId,
                          docId,
                          postedDate,
                          sectId,
                          sectNum,
                          sectName,
                          sectType,
                          prev1stLevelTitle,
                          prev2dLevelTitle,
                          sectText,
                          addMeta=None
                          ):
  metaDict = {SECTION_ID_FIELD: sectId,
              TITLE_1ST_LEVEL_FIELD: prev1stLevelTitle,
              TITLE_2D_LEVEL_FIELD: prev2dLevelTitle,
              SECT_NUM_FIELD: sectNum,
              TITLE_FIELD: sectName,
              LEVEL_FIELD: sectType,
              POSTED_DATE_KEY: postedDate,
              DOCKET_ID_KEY: docketId,
              DOCUMENT_ID_KEY: docId}
  if addMeta is not None:
    for key, val in addMeta.items():
      if key in metaDict:
        raise Exception('Bug: repeating key:' + str(key))
      metaDict[key] = val

  obj = {SPACY_TEXT_FIELD: sectText, SPACY_META_FIELD: metaDict}
  objStr = dumps(obj)
  outFile.write(objStr + '\n')

SectAnnotJSONL = namedtuple('SectAnnotJSONL', 'docketId, docId, sectId, sectType, \
                                              prev1stLevelTitle, prev2dLevelTitle, sectTitle, \
                                              startOffset, endOffset, sectText, type, modelPred')
SectAnnotJSONL.__new__.__defaults__ = (None,) * len(SectAnnotJSONL._fields)

def readSectAnnotJSONL(fileDir, fileName, docType, optText):

  with open(os.path.join(fileDir, fileName)) as f:
    for line in f:
      e = json.loads(line)
      if checkSpacyAccept(e):
        e = json.loads(line)
        text = e[SPACY_TEXT_FIELD]
        meta = e[SPACY_META_FIELD]
        docId = meta[DOCUMENT_ID_KEY]
        docketId = meta[DOCKET_ID_KEY]
        sectId = meta[SECTION_ID_FIELD]
        prev1stLevelTitle = meta[TITLE_1ST_LEVEL_FIELD]
        prev2dLevelTitle = meta[TITLE_2D_LEVEL_FIELD]
        sectTitle = meta[TITLE_FIELD]
        sectType = meta[LEVEL_FIELD]
        if docType == MASTER_JSON_TYPE_SPAN:
          if SPACY_SPAN_FIELD in e:
            for span in e[SPACY_SPAN_FIELD]:
              startOffset = int(span['start'])
              endOffset = int(span['end'])
              yield SectAnnotJSONL(docketId=docketId, docId=docId, sectId=sectId, sectType=sectType,
                                    prev1stLevelTitle=prev1stLevelTitle,
                                    prev2dLevelTitle=prev2dLevelTitle,
                                    sectTitle=sectTitle,
                                    startOffset=startOffset, endOffset=endOffset,
                                    sectText=text[startOffset:endOffset],
                                    type=span[SPACY_LABEL_FIELD].lower())
        elif docType == MASTER_JSON_TYPE_YESNO:
          # TODO extend optText
          if checkSpacyWholeDocAnnot(e, optText):
            startOffset = int(meta[START_OFFSET_FIELD])
            endOffset = int(meta[END_OFFSET_FIELD])
            yield SectAnnotJSONL(docketId=docketId, docId=docId, sectId=sectId, sectType=sectType,
                                    prev1stLevelTitle=prev1stLevelTitle,
                                    prev2dLevelTitle=prev2dLevelTitle,
                                    sectTitle=sectTitle,
                                    startOffset=startOffset, endOffset=endOffset,
                                    sectText=text,
                                    type=e[ACCEPT_FIELD][0]) # checkSpacyWholeDocAnnot verifies that the type is one-elem array

        else:
          raise Exception('Unsupported document type: %s' % docType)


def specialSectTitleFlag(titleArr, sectTitleTrainSet):
  for sectName in titleArr:
    if sectName.lower() in sectTitleTrainSet:
      return 1

  return 0

def getCompositeTitle(sectType, prev1stLevelTitle, prev2dLevelTitle, sectTitle, sep=' '):
  """Construct a composite title.

     :return  combined title (1st leel + 2d level + 3rd level if present).
  """

  titleArr = [prev1stLevelTitle, prev2dLevelTitle]
  if sectType == 3:
    titleArr.append(sectTitle)

  return sep.join(titleArr)