import re

import spacy

from utils.parse import replaceWhiteSpace


STOP_WORDS = frozenset(('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can',
                        'for', 'from', 'have', 'if', 'in', 'is', 'it', 'may',
                        'not', 'of', 'on', 'or', 'tbd', 'that', 'the', 'this',
                        'to', 'us', 'we', 'when', 'will', 'with', 'yet',
                        'you', 'your', 'inc',
                        "n't", "'s", "'ve", 'do'))


ALPHANUM_TOKENS = re.compile("^[a-zA-Z-_.0-9]+$")

def isGoodToken(s):
  return s and (ALPHANUM_TOKENS.match(s) is not None) and not s.startswith('-')

def filterTokens(tokLst):
  """Removes bad (mostly non alpha-numeric tokens)
  :param tokList a list of tokens

  :return a list of alphanum tokens.
  """
  return list(filter(isGoodToken, tokLst))

def filterSentToks(sentList):
  """Split sentences into tokens and filter each token list.

  :param a list of strings (sentences)
  :return a list of filtered tokens (as an array of tokens)
  """
  txt = ' '.join(sentList)
  return filterTokens(replaceWhiteSpace(txt).split())

def filterText(tokExtr, text):
  """Tokenizes the text and replaces gaps between tokens with spaces."""
  sentList = []
  for sent in tokExtr.getTokenizedSentences(text):
    if sent[2]:
      sentList.append(' '.join(sent[2]))

  return ' '.join(filterSentToks(sentList))

def splitTextIterator(tokExtractor, text, sentQty, sentSkip):
  """The procedure divides the text into overlapping pieces
  of text containing at most sentQty each. In addition, it
  splits the text into sentences and extracts tokens.
  Tokens can be also lemmatized by tokExtractor and subsequently
  merged into single chunk (where they are white space separated).
  Stop words may also be removed by the token extractor tokExtractor.
  The result may be lowercased (depending on the settings for the token
  extractor object).

  :param tokExtractor the object that does sentence segmentation,
                      token extraction, lemmatization, stopword, and punctuation removal.

  :param text         input text
  :param sentQty      a maximum number of sentences in a chunk
  :param sentSkip     a number of sentences to skip to generate the next chunk

  :return the function yields the following tuple: startOffset, endOffset, original text chunk, text chunk postprocessed.
  """
  chunkNum = 0
  sents = tokExtractor.getTokenizedSentences(text)

  if len(sents) == 0:
    yield 0, 0, '', ''
    return

  prevEnd = None

  for start in range(0, len(sents), sentSkip):
    end = min(start + sentQty, len(sents))
    assert(end > start)

    if prevEnd is not None and end == prevEnd:
      continue
    prevEnd = end

    startOffset = sents[start][0]
    endOffset = sents[end - 1][1]
    dispText = text[startOffset:endOffset]
    # Merge tokens to create a document for indexing
    indexTextArr = []
    for k in range(start, end):
      if sents[k][2]:
        indexTextArr.append(' '.join(filterSentToks(sents[k][2])))

    yield startOffset, endOffset, dispText, ' '.join(indexTextArr)

class TokenExtractor():
  """NOTE: this class uses a workaround for Spacy bug (see details below)."""
  def __init__(self, stopWords, lemmatize=True, lowerCase=True, ignorePunct=True,
               batchSize=50,
               threadQty=4):
    self.stopWords = set([s.lower() for s in stopWords])
    self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
    self.lemmatize = lemmatize
    self.lowerCase = lowerCase
    self.ignorePunct = ignorePunct
    self.threadQty = threadQty
    self.batchSize = batchSize

  def getLemma(self, tok):
    """This is a wrapper that works around against a Spacy lemmatization bug:
        it has a cost of carrying out lemmatization a second time (however,
        this should be cheap compared to POS tagging).
    """
    txt = str(tok.lemma_)
    return str(tok) if txt == '-PRON-' else txt

  def getTokensInternal2(self, res, doc):

    for t in doc:
      s = self.getLemma(t) if self.lemmatize else str(t)
      if not s.lower() in self.stopWords and \
        (not t.is_punct or not self.ignorePunct):
        res.append(s.lower() if self.lowerCase else s)

  def getTokensInternal1(self, docs):
    res = []
    for d in docs:
      self.getTokensInternal2(res, d)

    return res

  def getTokens(self, text):
    return self.getTokensInternal1(self.nlp.pipe([text],
                                                 batch_size=self.batchSize,
                                                 n_threads=self.threadQty))

  def getTokenizedSentences(self, text):
    """Retrieve a list of sentence annotations.

    :param text   input text

    :return a list of tuples (start sentence offset, end sentence offset, list of sentence tokens)
    """
    res = []
    docs = list(self.nlp.pipe([text],
                              batch_size=self.batchSize,
                              n_threads=self.threadQty))
    assert (len(docs) == 1)
    sents = docs[0].sents

    for oneSent in sents:
      sentToks = []
      self.getTokensInternal2(sentToks, list(oneSent.sent))

      start = oneSent.start_char
      end = oneSent.end_char
      res.append((start, end, sentToks))

    return res
