
from utils.roman import *
import numpy as np
import re

NL = '\n'

# We assume that there are no documents with more than 50 top-level sections
MAX_TOP_LEVEL_SECT_QTY = 50
INIT_2D_SECT_NUM = -1

BASIC_TAG_REGEX = re.compile('</?[a-zA-Z-_0-9]+/?>')
HREF_START_REGEX = re.compile('<a\\s+href[^>]+>')

LIST_OF_SUBJECTS = 'list of subjects'


# This is for extra checks only.
# don't use this crude function for parsing!
def removeBasicTags(line):
  return \
    HREF_START_REGEX.sub('',
      BASIC_TAG_REGEX.sub('', line))

# The original parser has a bug in identifying
# the end of the tag
#from html.parser import  HTMLParser
#class DocketHTMLParser(HTMLParser):

from utils.htmlParserFixed import HTMLParserFixed
class DocketHTMLParser(HTMLParserFixed):

  def __init__(self, docId):
    '''
    Constructor

    :param docId:  a document ID.
    '''
    # initialize the base class
    HTMLParserFixed.__init__(self)
    #HTMLParser.__init__(self)
    self.isInBody = False
    self.isInTitle = False
    self.wasInBody = False
    self.dataList = []
    self.docId = docId
    self.title = ''

  def handle_starttag(self, tag, attrs):
    tag = tag.lower()
    if tag == 'body':
      if self.isInBody or self.wasInBody:
        raise Exception('Repeating <body> tag in document %s' % self.docId)
      self.isInBody = True
      self.wasInBody = True
    # Most tags just need to be ignored
    elif tag in ['br'] and self.isInBody:
      self.dataList.append(' ')
    elif tag == 'title':
      if not self.isInBody and not self.wasInBody:
        self.isInTitle = True

  def handle_endtag(self, tag):
    tag = tag.lower()
    if tag == 'body':
      self.isInBody = False
    elif tag == 'title':
      self.isInTitle = False

  def handle_data(self, data):
    if self.isInTitle:
      self.title += data
    elif self.isInBody:
      self.dataList.append(data)

def isOneCharStr(line, char):
  '''Returns true if all string characters are the same'''
  return np.all(list(map(lambda x: x == char, list(line))))

def isTableRegularSepLine(line, minDashQty=10):
  '''
    Returns true if the line is a line/separator:
    it should contain only dashes (and not too few of them)
  '''
  line = line.strip()
  return len(line) >= minDashQty and isOneCharStr(line, '-')

def isTableStarSepLine(line, minStarQty=5, minSpaceQty=4):
  '''
  Returns true if the line looks like a separator with stars. Unlike the
    all-dash separator, such separator looks:
         * * * * *

  :param line: input string
  :param minStarQty: min # of stars to qualify
  :param minSpaceQty: min # of spaces to qualify

  :return: True if the line looks like a star-separator.
  '''
  lineNoSpace = re.sub('\\s+', '', line)
  res1 = (len(line) - len(lineNoSpace) >= minSpaceQty) and \
          len(lineNoSpace) >= minStarQty
  res2 = isOneCharStr(lineNoSpace, '*')
  return res1 and res2

def replaceWhiteSpace(line):
  return re.sub('\\s', ' ', line)

PAGE_NUM_REGEX_OLD=re.compile('^\\s*\\[\\[page\\s+[0-9]+\\]\\]\\s*$')

# Page number related regexps
PAGE_NUM_REGEX=re.compile('^\\s*\\[\\[page\\s+([0-9]+)\\]\\]\\s*$')
PAGE_NUM_RANGE_REGEX1=re.compile('^\\s*\\[page\\s+([0-9]+)\\]\\s*$')
PAGE_NUM_RANGE_REGEX2=re.compile('^\\s*\\[pages?\\s+([0-9]+)\\-([0-9]+)\\]\\s*$')

def toIntNoExcept(v):
  '''Try to convert to int, but fail silently without an exceptoin.

  :param input value to convert
  :return a converted value or None if conversion fails
  '''
  try:
    return int(v)
  except Exception:
    return None

def isPageNumLine(line):
  '''Check if the line is a page number line. If this is the case,
    obtain the page number.

  :param line:
  :return: a tuple (True/False depending on whether a page number is detected,
                    the page or None if the line doesn't have a page number)
  '''
  x = PAGE_NUM_REGEX.match(line.lower())
  if x is None:
    return False, None

  pgNum=toIntNoExcept(x[1])
  return pgNum is not None, pgNum

def isPageRangeNum(line):
  '''Check if the line is a page-range line. If true,
     retrieve the starting and the ending page of this range (both INCLUSIVE).

    :param line: input line
    :return: a triple (a boolean flag, None or the starting page, None or the ending page)
  '''

  line = line.lower()
  x = PAGE_NUM_RANGE_REGEX1.match(line)

  if x is not None:
    start = end = toIntNoExcept(x[1])
    return start is not None, start, end

  x = PAGE_NUM_RANGE_REGEX2.match(line)

  if x is not None:
    start = toIntNoExcept(x[1])
    end = toIntNoExcept(x[2])

    if start is not None and \
       end is not None and \
       start <= end:
      return True, start, end

  return False, None, None

ALPHA_NUM_REGEX=re.compile(r'[a-zA-Z-0-9\']')

def isPageNumLineOld(line):
  return PAGE_NUM_REGEX_OLD.match(line.lower()) is not None

# Although section names nearly always start with a space,
# let's bit a bit permissive and allow for an extra leading space,
# which sometimes happens. Along the same lines, there might
# be a space before dot
SECTION_NAME_REGEX=re.compile(r'^\s?([A-Z]+|[a-z]+|[0-9]+)\s?[.]\s.*$')

# If it is not a proper Roman numeral,
# it should be a single either a single letter or a sequence of digits
# followed by a dot and a space
SECTION_NAME_DIGIT_REGEX=re.compile(r'^\s?([0-9]+)\s?[.]\s.*$')
SECTION_NAME_ROMAN_CAPITAL_REGEX=re.compile(r'^\s?[A-Z]\s?[.]\s.*$')

def getSectionNumAndName(line):
  '''
  Given that a line is a numbered section name, return section number (as a string)
  and a name.

  :param line: input line
  :return: a tuple: section number section name.
  '''
  line = replaceWhiteSpace(line.strip())
  i = line.find('.')
  return (line[0:i].strip(), line[i+1:].strip()) if i >= 0 else (None, line)


def secondLevelStrNumToNum(numStr):
  '''Try to convert something looking like a 2d section
  name to a number.'''

  if not numStr:
    return None

  for c in numStr:
    if c != numStr[0] or c < 'A' or c > 'Z':
      return None

  return ord(numStr[0]) - ord('A') + (ord('Z') - ord('A') + 1) * (len(numStr) - 1)

def isNumberedSectionName(line):
  '''
  Check if a line looks like a numbered section name.

  :param line: input line
  :return: return True if the line looks like a numbered section name.
  '''
  line = replaceWhiteSpace(line)
  if SECTION_NAME_REGEX.match(line) is None:
    return False
  line = line.strip()
  if SECTION_NAME_DIGIT_REGEX.match(line) is not None:
    return True
  num, _ = getSectionNumAndName(line)
  num = num.upper()
  if secondLevelStrNumToNum(num) is not None:
    return True
  return roman_to_int_wrapper(num) is not None


def isRomanCapitalSectionName(line):
  '''
  Check if a line looks like a second-level Roman-number section name.
  Note it is sometimes not possible to distinguish from a top-level section
  name. E.g. I. ... can be used in both.

  :param line: input line
  :return: True if condition is matched
  '''
  return SECTION_NAME_ROMAN_CAPITAL_REGEX.match(line) is not None

def isTableStart(startLineNum, lines, maxCaptionLineQty=8):
  '''
  Check if the table starts at a given line number.

  :param startLineNum: starting line number
  :param lines: array of lines/strings
  :param maxCaptionLineQty maximum # of caption lines
  :return: a tuple: boolean flag, the first line number beyond the header
  '''

  if startLineNum + 2 < len(lines):

    hasStandCaption =  lines[startLineNum].strip() == '' and \
          lines[startLineNum+1].strip().startswith('Table ')

    if hasStandCaption:
      ln = startLineNum + 2
      while ln < min(startLineNum + maxCaptionLineQty, len(lines)):
        if isTableRegularSepLine(lines[ln]):
          return True, ln + 1
        ln += 1

    if isTableRegularSepLine(lines[startLineNum+1]):
      ln = startLineNum + 2

      while ln < min(len(lines), startLineNum + 2 + maxCaptionLineQty):
        if isTableRegularSepLine(lines[ln]):
          return True, ln + 1 # Hurray the end of the header is found
        # Most headers start with a space and contain some alpha-numeric characters
        # However, this is not universally true
        #if not lines[ln].startswith(' '):
        #  return False, startLineNum

        line = lines[ln].strip()
        # This heuristics fails sometimes, but not frequently, e..g, for
        # requirement (include checklist     (and/or RCRA       Analogous state
        # in EPA/R08/RCRA/2009/0341/EPA-R08-RCRA-2009-0341-0002.html
        if ALPHA_NUM_REGEX.match(line) is None:
          return False, startLineNum

        ln = ln + 1

  return (False, startLineNum)

def extractTables(docText, maxEmptySkipQty=4):
  '''
  Attempt to extract tables.

  :param docText:
  :param maxEmptySkipQty: a maximum number of empty lines in the table

  :return: a tuple: text (as an array of lines) with removed tables, a list of tables
  '''
  docText = docText.replace('\r', '')
  docLines = docText.split(NL)

  end = len(docLines)
  start = 0

  resText = []
  resTables = []

  while start < end:
    tabStart, nextLine1 =  isTableStart(start, docLines)
    if not tabStart:
      resText.append(docLines[start])
      start += 1
    else:
      # Here we need to scan to the end of the table
      oneTab = []

      while start < end:

        line = docLines[start]

        oneTab.append(line)
        line = line.strip()

        # In a vast majority of  cases tables don't have empty lines
        # inside but there are a few exceptions to this rule
        if line != '':
          start += 1 # Just move to the next line

        else: # The rule for the code below:
          # i) it should advance the index variable start and memorize
          #    lines inside the table
          # ii) if we reached the end of the table, the index variable
          #     should point to the first line after the table


          # Possible reasons for the empty line:
          # i) page number
          # ii) there's another separating line afterwards
          # iii) end of table

          # However, if something looks like a section name
          # we better stop processing the table immediately,
          # otherwise we have a risk of missing some sections
          if start + 1 < end and isNumberedSectionName(docLines[start + 1]):
            start += 1
            break

          curr = start

          eofTable = True

          while curr < min(start + maxEmptySkipQty, end):

            currLine = docLines[curr].strip()

            if isTableRegularSepLine(currLine):
              # EPA_FRDOC_0001-16086.html has the empty lines between two regular separating lines
              # EPA_FRDOC_0001-19711.html has the empty line between the star separating line
              # Very rarely there can be two and three empty lines (see the checks below)
              oneTab.append(currLine)
              eofTable = False
              curr += 1 # Skipping one of the separating lines, but not finishing scanning the table
              break

            if currLine != '':
              break

            curr += 1

          start = curr
          if eofTable:
            break

      # Let's add an extra line: otherwise the following
      # paragraph will not be properly separated.
      # This would be especially harmful, if the paragraph
      # is a section name.
      resText.append('')
      resTables.append(NL.join(oneTab))


  return resText, resTables

def removePageNumbers(docText, docId):
  '''
  Removes page numbers while checking for the numbering consistency:
  this is possible, b/c
  i) In most cases the range of pages is given in the pre-amble.
  ii) Page numbers is a monotonically increasing sequence.

  :param docText: document text
  :param docId: document id
  :return: a tuple (document text split into lines, but with pages removed,
                    page numbers)
  '''
  srcLines = docText.split(NL)

  pgRangeNumLine = None
  pgStart = None
  pgEnd = None
  for i in range(len(srcLines)):
    flag, s, e = isPageRangeNum(srcLines[i])
    if flag:
      if pgRangeNumLine is not None:
        raise Exception('Duplicate page-range definition in document %s, encountered in lines %d and %d'
                        % (docId, pgRangeNumLine + 1, i + 1))
      pgRangeNumLine = i
      pgStart, pgEnd = s, e


  # Let's remove page numbers
  start = 0
  end = len(srcLines)
  resLines = []
  prevPageNum = None
  pageNums = []

  while start < end:

    line = srcLines[start]

    flag, pgNum = isPageNumLine(line)

    if not flag:

      resLines.append(line)
      start += 1

    else:
      # It is crucial to remove new lines previous empty lines
      while resLines and resLines[-1].strip() == '':
        resLines.pop()

      # check for consistency
      if prevPageNum is not None:
        if pgNum != prevPageNum + 1:
          raise Exception('Non continuous numbering in document %s previous pg. # %d the current pg. # is %d'
                          % (docId, prevPageNum, pgNum))

      if pgStart is not None:
        if pgNum < pgStart - 1 or pgNum > pgEnd + 1:
          raise Exception('Page number %d in document %s is outside the preamble declared rage %d-%d' %
                          (pgNum, docId, pgStart - 1 , pgEnd + 1))
      prevPageNum = pgNum
      pageNums.append(pgNum)

      start += 1  # skip start + 1, which a page number

      # Skip empty lines after the page number
      # but only if there's no following section name
      while start < end and srcLines[start].strip() == '':
        if start + 1 < end and isNumberedSectionName(srcLines[start+1]):
          break
        start += 1

  return resLines, pageNums



def removePageNumbersOld(docText):
  '''
  Removes page numbers (older versions).

  :param docText: document text.
  :return: a tuple (document text split into lines, but with pages removed,
                    page lengths)
  '''
  srcLines = docText.split(NL)

  # Let's remove page numbers
  start = 0
  end = len(srcLines)
  resLines = []
  prevPageStart = 0
  pageLens = []

  while start < end:

    line = srcLines[start]
    if line != '' or \
      start + 1 == end or \
      not isPageNumLineOld(srcLines[start + 1]):

      resLines.append(line)
      start += 1

    else:

      pageLens.append(start - prevPageStart)

      start += 2  # skip start + 1, which is empty, and start + 2 which is a page number
      # Skip empty lines after the page number
      while start < end and srcLines[start].strip() == '':
        start += 1

      prevPageStart = start

  # Last page
  if prevPageStart < start:
    pageLens.append(start - prevPageStart)

  return resLines, pageLens

def scanParagraph(lines, currLine, keepNL=False):
  '''
  This function scans text of a paragraph or of a section name.
  A paragraph/section name ends when we encounter:
  1) An empty line.
  2) A line that starts with a space symbol
  3) A section name
  4) A table ruler/separator

  :param lines:   an array of document lines
  :param currLine: a current line where the paragraph starts
  :param keepNL: if true, preserve the carriage return

  :return: a tuple : first line after the paragraph, concatenated paragraph text.
  '''
  end = len(lines)
  if currLine >= end:
    return end, ''

  paraText = lines[currLine].rstrip()
  currLine += 1

  while currLine < end:

    line = replaceWhiteSpace(lines[currLine])
    if line.startswith(' ') or line.strip() == '' or \
      isNumberedSectionName(line) or \
      isTableRegularSepLine(line) or isTableStarSepLine(line):
      break

    if not keepNL:
      paraText += ' ' + line.rstrip()
    else:
      paraText += '\n' + line.rstrip() + '\n'

    currLine += 1

  return currLine, paraText


def parseDocketStep1(docHtml, docId):
  '''
    Carry out the initial step of parsing:
    1. extract text and title.
    2. remove special start-separator lines
    3. replace white-spaces with a regular space

  :param docHtml: document HTML
  :param docId: document ID
  :return: a tuple: HTML title (but not document title), text without HTML tags
  '''
  parser = DocketHTMLParser(docId)
  parser.feed(docHtml)

  lines = parser.dataList
  for i in range(len(lines)):
    if isTableStarSepLine(lines[i]):
      lines[i] = ''

  # It's important to join things here on the empty line
  docText = ''.join(lines)

  return parser.title, docText

def parseDocketStep2(docText, docId):
  '''
    Carry out the second parsing step :
    1. Remove page numbers
    2. Remove tables.

  :param docText: document text produced by the function parseDocketStep1
  :param docId: document ID
  :return: a tuple with a processed text and a list of tables
  '''
  lines, _ = removePageNumbers(docText, docId)
  docText = NL.join(lines)

  resLines, tabList = extractTables(docText)

  return NL.join(resLines), tabList

def parseDocketStep3(docText, minTocQty=3):
  '''
     1. Remove separator lines
     2. Extract table of contents

  :param docText: input text.
  :param minTocQty: the minimum number of sections to expect in the table of
                    contents (TOC).
  :return: a tuple: processed text, table of contents (TOC).
  '''

  lines = docText.split(NL)

  # Group lines into blocks separated by one or more newline

  toc = ''

  resLines = []

  start = 0
  end = len(lines)

  while start < end:
    line = replaceWhiteSpace(lines[start])
    if isTableRegularSepLine(line):
      line = ''
    if line.strip() != '' and not isNumberedSectionName(line) \
        or toc != '': # if we have found TOC, we just scan to the end of the doc
      start += 1
      resLines.append(line)
    else:
      # This might be a start of a table of contents
      # preceded by a number of empty lines
      tocLines = []

      tocStart = start
      while tocStart < end and lines[tocStart].strip() == '':
        tocStart += 1

      tocCurr = tocStart

      if tocCurr < end:
        expRomanNum = 1

        prevSectId = None

        while tocCurr < end:
          #
          # Basic algorithm:
          #
          # 1. Scan what seems to be a TOC till the first fully empty line occurs
          # 2. If the line starts with no space (and there is no preceding Roman numeral
          #    in the previous lines, which will be accounted for at a different step of the
          #    algorithm by reading all such lines following a Roman-numeral section name),
          #    2.1 this lines must start with a Roman numeral followed by a dot.
          #    2.2 Roman numerals must be all properly ordered
          # 3. If the line starts with the space,
          # 4. If we encountered a section name,
          #    read the section name until ends (it has the same ending rules
          #    as a paragraph).
          # 5. TOC must have at least minTocQty lines
          #
          line = replaceWhiteSpace(lines[tocCurr])
          if line.strip() == '':
            break # Ok, end of TOC!

          if tocCurr == tocStart and line.startswith(' '):
            # Not a valid line: we expect a Roman-numeral numbered section name here
            tocLines = [] ; break

          if not isNumberedSectionName(line.strip()):
            tocLines = [] ; break

          #print('!!', line)

          if line.startswith(' '):
            prevSectId, _ = getSectionNumAndName(line.strip())
          else:
          # Possibly a top-level section name
            sectId, sectName = getSectionNumAndName(line)
            num = roman_to_int_wrapper(sectId)
            if num is not None \
              and sectId == sectId.upper() \
              and num == expRomanNum:
              expRomanNum += 1  # Increment the counter
            else:
              # Sometimes these second-level sections just lack indent :
              # however if the current section ID is a continuation of the previous one
              # we are fine
              if prevSectId is not None and \
                len(prevSectId) == len(sectId) == 1 and \
                 ord(prevSectId) + 1 == ord(sectId):
                pass
              else:
                tocLines = [] ; break

          # Complete the header by appending additional lines.

          tocCurr, sectName = scanParagraph(lines, tocCurr)

          tocLines.append(sectName)

      if not tocLines or len(tocLines) < minTocQty:

        resLines.append(lines[start])
        start += 1

      else:
        toc = NL.join(tocLines)

        start = tocCurr

  return NL.join(resLines), toc

def parseDocketStep4(docText, toc='', debugPrint=False):
  '''The final parsing step that extracts sections, divides the
  text into paragraphs and assigns a section to each paragraph.

  :param docText: input text
  :param toc: if not None, we will check that section names appear in the TOC.

  :return: list of tuples : section name, list of section paragraphs.
                            important note: the section name can be
                            sometimes empty (unnamed sections).
  '''
  tocSet = set()
  if toc is None:
    toc = ''

  if toc != '':
    for tln in toc.split(NL):
      tocSet.add(tln.strip().lower())


  res = []

  start = 0
  docLines = docText.split(NL)
  end = len(docLines)

  res.append( ('', []) )

  while start < end:
    while start < end and docLines[start].strip() == '':
      start += 1

    paraStart = start
    start, paraText = scanParagraph(docLines, start, keepNL=False)
    # Now start points to the first line after the paragarph

    if paraStart > 0 and docLines[paraStart-1].strip() == '' and \
      start < end and \
      (docLines[start].strip() == '' or docLines[start].startswith(' ')) and \
      (isNumberedSectionName(paraText) or paraText.lower().startswith(LIST_OF_SUBJECTS)):
      sectName = replaceWhiteSpace(paraText).strip()
      if toc != '' and sectName.lower() not in tocSet:
        if debugPrint:
          print('Section not found in the TOC: %s' % sectName)
      res.append( (sectName, []) )
    else:
      res[-1][1].append(paraText)

  return res

def findSectType(sectNums, maxTopLevelSectQty,
                 curr = 0, prevTopSectNum = 0,
                 prev2dSectNum = INIT_2D_SECT_NUM,
                 debugPrint=True):
  sectQty = len(sectNums)
  if curr >= sectQty:
    return [ [] ] # It shouldn't be an empty list, but rather
                  # a list containing an empty list!
  numStr=sectNums[curr]

  # 2d-level section can start before we saw the top-level section:
  # e.g., EPA/R08/RCRA/2009/0341/EPA-R08-RCRA-2009-0341-0002.html
  # simply has no top-level Roman-numeral sections.
  res = []

  canBe2dLevel = False
  canBeTopLevel = False

  if debugPrint:
    print('###', curr, numStr)

  if numStr is not None: # Some sections are nameless
    numVal = secondLevelStrNumToNum(numStr)
    if numVal is not None:
      canBe2dLevel = True

      if numVal == prev2dSectNum + 1 or \
        (numVal == 1 and prev2dSectNum == INIT_2D_SECT_NUM): # Quite often sub-section A is missing
        if debugPrint:
          print('###', curr, '2d')
        recRes = findSectType(sectNums, maxTopLevelSectQty,
                              curr + 1, prevTopSectNum,
                              numVal,
                              debugPrint)
        if debugPrint:
          print('###', curr, '2d ret: #', len(recRes))
        for e in recRes:
          res.append([2] + e)

    numConv = roman_to_int_wrapper(numStr)
    # The top-level section can be only all-capital Roman numeral
    # (sometimes though we don't have this top-level numeration and
    #  chapters are numbered using A. B. C. etc..
    # # e.g., EPA/R08/RCRA/2009/0341/EPA-R08-RCRA-2009-0341-0002.html
    if numStr == numStr.upper() and \
      numConv is not None:
      canBeTopLevel = True

      if numConv is not None and \
        (numConv == prevTopSectNum + 1):
        if debugPrint:
          print('###', curr, '1st')
        recRes = findSectType(sectNums, maxTopLevelSectQty,
                              curr + 1, numConv,
                              INIT_2D_SECT_NUM,
                              debugPrint)
        if debugPrint:
          print('###', curr, '1st ret: #', len(recRes))
        for e in recRes: # Second-level sections must start from scratch!
          res.append([1] + e)

  if numStr is None:
    canBeTopLevel = True
    if debugPrint:
      print('###', curr, '1st noname')
    # We restart top-level numeration after the List of subjects section
    recRes = findSectType(sectNums, maxTopLevelSectQty,
                          curr + 1, 0,
                          INIT_2D_SECT_NUM,
                          debugPrint)
    if debugPrint:
      print('###', curr, '1st noname', len(recRes))
    for e in recRes:  # Second-level sections must start from scratch!
      res.append([1] + e)

  # Only if the section number cannot be 1d or 2d level
  # do we consider it to be a 3+ level section
  # Please, note that a section looks 1st or 2d level,
  # but it doesn't continue proper enumeration,
  # it still will *NOT* be treated as a 3+ level sections.
  # If the latter were not true, we would have had a way
  # to skip sections with invalid enumeration and, consequently,
  # generate improper section type assignment
  if not canBeTopLevel and not canBe2dLevel:
    if debugPrint:
      print('###', curr, '2+')
    recRes = findSectType(sectNums, maxTopLevelSectQty,
                          curr + 1, prevTopSectNum,
                          prev2dSectNum,
                          debugPrint)
    if debugPrint:
      print('###', curr, '2+ ret: #', len(recRes))
    for e in recRes:
      res.append([3] + e)

  return res

def partialParse(fileName, debugPrint=False):

  docHtml = ''.join(list(open(fileName)))

  _, text1 = parseDocketStep1(docHtml, fileName)
  text2, _ = parseDocketStep2(text1, fileName)
  text3, toc = parseDocketStep3(text2)
  #print(text3)
  if debugPrint:
    print('****DETECTED TOC****:')
    print(toc)

  sectNums  = []
  sectNames = []
  sectParas = []

  for sectNameLine, paraList in parseDocketStep4(text3, toc, debugPrint):
    num, name = getSectionNumAndName(sectNameLine)
    sectNums.append(num)
    sectNames.append(name)
    sectParas.append(paraList)

  if debugPrint:
    print(fileName)
    print(sectNums)
    print(sectNames)

  return sectNums, sectNames, sectParas

def fullParse(fileName, maxTopLevelSectQty = MAX_TOP_LEVEL_SECT_QTY, debugPrint=False):
  '''Carry out all the parsing stages of the document, plus identify
     which section names are top-level and which ones are second-level.
     Other section names are not taken into account.

    :param fileName: the name of the input file.
  '''

  sectNums, sectNames, sectParas = partialParse(fileName, debugPrint)

  res = findSectType(sectNums, maxTopLevelSectQty, debugPrint=debugPrint)

  if not res:
    raise Exception('Cannot generate a section type assignment for:\n%s'
                    % (fileName))

  # In rare cases it is not possible to generate a unique assignment
  if len(res) > 1:
    print('WARNING: Cannot generate a single unique section type assignment for %s number of assignments %d, choosing the first one'
                    % (fileName, len(res)))

  sectTypes = res[0]

  return sectNums, sectTypes, sectNames, sectParas


def fullParseSectionIterator(fileName, parSep = '\n'):
  sectNums, sectTypes, sectNames, sectParas = fullParse(fileName)

  print('Processing: %s # of sections: %d' % (fileName, len(sectNums)))

  for sectId in range(len(sectNums)):
    sectTitle = sectNames[sectId]
    sectType = sectTypes[sectId]

    if sectType == 1:
      prev1stLevelTitle = sectTitle
      prev2dLevelTitle = ''
    elif sectType == 2:
      prev2dLevelTitle = sectTitle

    sectNum = sectNums[sectId]
    sectText = parSep.join(sectParas[sectId])

    # sectType is section level
    yield sectId, sectNum, sectType, sectTitle, prev1stLevelTitle, prev2dLevelTitle, sectText





