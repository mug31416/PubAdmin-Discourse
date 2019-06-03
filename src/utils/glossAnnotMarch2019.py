import json
import sys

sys.path.append('.')

from utils.misc import *
from utils.spacyAnnot import *
from utils.io import *

MIN_ANNOT_OVERLAP = 0.8

EXAMPLE_DOCUMENTS = ["EPA-HQ-OAR-2012-0133-0067.txt","EPA-R04-OAR-2017-0436-0012.txt","EPA-HQ-SFUND-1990-0011-0136.txt"]
TEST_DOCUMENTS = ["EPA-HQ-OLEM-2016-0177-0089.txt","EPA-R09-OAR-2015-0622-0063.txt",
                  "EPA-HQ-TRI-2011-0174-0011.txt","EPA-HQ-OAR-2009-0299-0070.txt"]
EXPERT_OWNER = "5c491ecbee951f365779e552"
EXCLUDED_OWNERS = ["5ad5082accf5e87c942303c4","58a0bf4f8424bd4f65e2be57","5c5c769cee951f365779ead2"]
ANNO_TYPES = ['COMMENT_DISCUSSION','CHANGE_IN_RULE','NO_CHANGE_IN_RULE']

EXPERT_OWNER_EXCLUDE_DOCS = ["EPA-HQ-OAR-2002-0056-6739.txt"]

JSON_MARKER = '###JSON###'
META_MARKER = '##META##'

ERROR_FILE_NAME_PREF='error_annot'

class LineIter:

  def eof(self):
    return self.cur >= len(self.text)

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

class SectionProps:

  def __init__(self, tokExtr, sectTitleTrainSet, startOffset, setJSON, sectText, sectAnnot):
    self.startOffset = startOffset
    self.sectId = setJSON[SECTION_ID_FIELD]
    self.sectType = setJSON[LEVEL_FIELD]
    self.prev1stLevelTitle = setJSON[TITLE_1ST_LEVEL_FIELD]
    self.prev2dLevelTitle = setJSON[TITLE_2D_LEVEL_FIELD]

    self.titleSectFlag = specialSectTitleFlag([self.prev1stLevelTitle, self.prev2dLevelTitle], sectTitleTrainSet)
    self.sectCompTitle = getCompositeTitle(self.sectType, self.prev1stLevelTitle, self.prev2dLevelTitle, setJSON[TITLE_FIELD])

    self.sectAnnot = sectAnnot

    self.sectSents = []

    for sentStart, sentEnd, _ in tokExtr.getTokenizedSentences(sectText):
      self.sectSents.append((sentStart, sentEnd, sectText[sentStart:sentEnd]))


def readGlossAnnots(tokExtr, glossJSONFileName, sectTitleTrainJSONLFileName, docSetJSONFileName, exploreAgreement = False):
  glossJSON = readJson(glossJSONFileName)
  docSetJSON = readJson(docSetJSONFileName)
  sectTitleTrainSet = loadTrainTitles(sectTitleTrainJSONLFileName)

  docSet = docSetJSON["documents"].keys()

  error_file_n = 0

  for _, doc_json in glossJSON['documents'].items():

    props = []

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

        textLineIter.goNext()
        if textLineIter.eof():
          break

        startOffset = textLineIter.getStartOffset()

        while not textLineIter.eof() and not textLineIter.getLine().startswith(META_MARKER):
          textLineIter.goNext()

        endOffset = textLineIter.getStartOffset()
        sectText = text[startOffset:endOffset]

        # Here do something with text from start to end

        sectAnnot = []

        for _, e in annots.items():

          owner = e['owner']

          if owner in EXCLUDED_OWNERS:
            print("@@@@@Excluded owner ", owner)
            continue

          if exploreAgreement is False:

            if doc_name in TEST_DOCUMENTS and owner != EXPERT_OWNER:
              print("@@@@@Test document, defer annotation to expert")
              continue

            if doc_name in EXAMPLE_DOCUMENTS and owner != EXPERT_OWNER:
              print("@@@@Example document, defer annotation to expert")
              continue

          if doc_name in EXPERT_OWNER_EXCLUDE_DOCS and owner == EXPERT_OWNER:
            print(f"@@@@Exclude {doc_name} which was abandoned by expert, but erroneously remained in data")
            continue

          if e['type'] in ANNO_TYPES:
            annotStart, annotEnd = e['span']

            if annotStart >= startOffset and annotEnd <= endOffset:
              sectAnnot.append( (annotStart - startOffset, annotEnd - startOffset, owner, e['type']) )

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

                sectAnnot.append((annotStart - startOffset, annotEnd - startOffset, owner, e['type']))


        #print(sectAnnot)
        for ast, aend, _ , _ in sectAnnot:
          astOrig = ast + startOffset
          aendOrig = aend + startOffset
          origAnnotText = text[astOrig:aendOrig]

          sectAnnotText = sectText[ast:aend]
          if (origAnnotText != sectAnnotText):
            print(origAnnotText)
            print('------------')
            print(sectAnnotText)
            print('------------')
            raise Exception('Bug: annotation offset mismatch!')

        props.append(SectionProps(tokExtr, sectTitleTrainSet, startOffset, setJSON, sectText, sectAnnot))

    yield docketId, doc_name, props, doc_json




