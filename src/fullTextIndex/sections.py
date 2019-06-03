from fullTextIndex.commonForIndex import *
from fullTextIndex.textProcForIndex import *
from utils.misc import *

SECTION_TEXT_FIELD = "sectionText"
SECTION_TEXT_TITLE_INDEX_FIELD = "sectionTitleTextIndex"

SECTION_STORED_FIELDS = [CHUNK_ID_FIELD, DOCKET_ID_FIELD, DOCUMENT_ID_KEY,
                         SECTION_ID_FIELD,
                         TITLE_1ST_LEVEL_FIELD, TITLE_2D_LEVEL_FIELD, TITLE_FIELD,
                         LEVEL_FIELD, SECT_NUM_FIELD,
                         SECTION_TEXT_FIELD,
                         START_OFFSET_FIELD, END_OFFSET_FIELD,
                         POSTED_DATE_KEY]

def indexOneSectionDoc(tokExtr,
                sectWriter,
                postedDate,
                docketId, docId,
                sectId, sectNum, sectType, sectName, prev1stLevelTitle, prev2dLevelTitle, sectText,
                sentQty, sentSkip):

  title = prev1stLevelTitle
  if prev2dLevelTitle != '':
    title += ' / ' + prev2dLevelTitle
  if sectType == 3 and sectName != '':
    title = title + ' / ' + sectName
  # Clean-up title
  filteredTitle = filterText(tokExtr, title)

  # Let's index separate chunks
  chunkId = 0
  for startOffset, endOffset, docChunkDispTxt, docChunkIndexTxt in splitTextIterator(tokExtr, sectText, sentQty, sentSkip):
    # We want an index that includes both the complete doc title
    # and the index chunk text
    docChunkIndexTxt = filteredTitle + ' ' + docChunkIndexTxt

    if False:
      print('--------------')
      print(docIdChunk, startOffset, endOffset)
      print('--------------')
      print(docChunkIndexTxt)
      print('--------------')
      print(docChunkDispTxt)
      print('==============')

      chunkId += 1

    docChunk = Document()

    # Indexable fields
    indexDataDict = {CHUNK_ID_FIELD : IndexFieldInfo(chunkId),
                    DOCKET_ID_FIELD : IndexFieldInfo(docketId),
                    DOCUMENT_ID_KEY:  IndexFieldInfo(docId),
                    SECTION_TEXT_TITLE_INDEX_FIELD : IndexFieldInfo(docChunkIndexTxt, fieldType='text', stored=False)}

    addIndexFields(docChunk, indexDataDict)

    # Just (nearly unprocessed) text
    metaDict = {SECTION_ID_FIELD: sectId,
                TITLE_1ST_LEVEL_FIELD: prev1stLevelTitle,
                TITLE_2D_LEVEL_FIELD: prev2dLevelTitle,
                TITLE_FIELD: sectName,
                LEVEL_FIELD: sectType,
                SECT_NUM_FIELD : sectNum,
                SECTION_TEXT_FIELD : docChunkDispTxt,
                START_OFFSET_FIELD : startOffset,
                END_OFFSET_FIELD : endOffset,
                POSTED_DATE_KEY: postedDate,
                }

    for fn in SECTION_STORED_FIELDS:
      assert(fn in metaDict or fn in indexDataDict)

    addStoredFields(docChunk, metaDict)

    sectWriter.addDocument(docChunk)
