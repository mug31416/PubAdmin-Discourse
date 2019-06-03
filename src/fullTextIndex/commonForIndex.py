import os, re

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader, IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.document import Document, Field, IntPoint, StringField, TextField, StoredField
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.queryparser.classic import QueryParser

# BM25 coefficients
K1 = 1.2
B = 0.75

# By default we index passage that are 5 sentences long and have zero overlap
DEFAULT_PASS_SENT_QTY=5
DEFAULT_PASS_SENT_SHIFT=5

RAM_BUFFER_SIZE = 1024 * 8 # 8 GB

COMMIT_QTY = 100

def initLucene(initRamSize):
  lucene.initVM(vmargs=['-Djava.awt.headless=true -Xms%dm' % initRamSize])

def createIndexSearcher(indexDir):
  directory = DirectoryReader.open(FSDirectory.open(Paths.get(indexDir)))
  searcher = IndexSearcher(directory)
  similarity = BM25Similarity(K1, B)
  searcher.setSimilarity(similarity)
  return searcher


def createIndexWriter(indexDir):
  if not os.path.exists(indexDir):
    os.mkdir(indexDir)
  directory = FSDirectory.open(Paths.get(indexDir))
  config = IndexWriterConfig(WhitespaceAnalyzer())
  #config = config.setRAMBufferSizeMB(ramBufferSize)
  config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
  return IndexWriter(directory, config)

class IndexFieldInfo:
  """Properties of an indexable String or Text field"""
  def __init__(self, value, fieldType='string', stored=True):
    if fieldType != 'text' and fieldType != 'string':
      raise Exception('Unsupported field type %s (only string and text are expected)' % fieldType)

    self.value = value
    self.type = fieldType
    self.stored = stored

def addIndexFields(doc, dataDict):
  for key, info in dataDict.items():
    if type(info) != IndexFieldInfo:
      raise Exception('Wrong field info type: %s expecting %s' % (str(type(info)), str(type(IndexFieldInfo))))
    storeFlag = Field.Store.YES if info.stored else Field.Store.NO
    fieldType = StringField if info.type == 'string' else TextField
    doc.add(fieldType(key, str(info.value), storeFlag))


def addStoredFields(doc, metaDict):
  for key, value in metaDict.items():
    value = '' if value is None else value
    doc.add(StoredField(key, value))

def extractFields(doc, fieldNames):
  res = {}
  for fn in fieldNames:
    res[fn] = doc.get(fn)
  return res

