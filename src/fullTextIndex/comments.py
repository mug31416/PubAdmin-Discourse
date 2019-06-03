from fullTextIndex.commonForIndex import *
from fullTextIndex.textProcForIndex import *

from utils.misc import *


COMMENT_TEXT_FIELD = "commentText"
COMMENT_TITLE_FIELD = TITLE_FIELD

COMMENT_TEXT_TITLE_INDEX_FIELD = "commentTitleTextIndex"

COMMENT_STORED_FIELDS = [CHUNK_ID_FIELD, DOCKET_ID_FIELD, DOCUMENT_ID_KEY,
                         COMMENT_TITLE_FIELD, COMMENT_TEXT_FIELD, START_OFFSET_FIELD, END_OFFSET_FIELD]

def indexOneCommentDoc(tokExtr,
                       commentWriter,
                       docketId, docId,
                       title, text,
                       sentQty, sentSkip):
  # Let's index separate chunks
  # Clean-up title
  filteredTitle = filterText(tokExtr, title)

  chunkId = 0
  for startOffset, endOffset, docChunkDispTxt, docChunkIndexTxt in \
      splitTextIterator(tokExtr, text, sentQty, sentSkip):

    # We want an index that includes both the complete doc title
    # and the index chunk text
    docChunkIndexTxt = filteredTitle + ' ' + docChunkIndexTxt

    #print(docIdChunk, docChunkDispTxt)

    chunkId += 1

    # Also need to store the original unlemmatized comment!!!

    docChunk = Document()

    # Indexable fields
    indexDataDict = {CHUNK_ID_FIELD : IndexFieldInfo(chunkId),
                    DOCKET_ID_FIELD : IndexFieldInfo(docketId),
                    DOCUMENT_ID_KEY:  IndexFieldInfo(docId),
                    COMMENT_TEXT_TITLE_INDEX_FIELD : IndexFieldInfo(docChunkIndexTxt, fieldType='text', stored=False)}

    addIndexFields(docChunk, indexDataDict)

    # Just (nearly unprocessed) text
    dataDict = {COMMENT_TITLE_FIELD : title,
                COMMENT_TEXT_FIELD : docChunkDispTxt,
                START_OFFSET_FIELD : startOffset,
                END_OFFSET_FIELD : endOffset}

    for fn in COMMENT_STORED_FIELDS:
      assert(fn in dataDict or fn in indexDataDict)

    addStoredFields(docChunk, dataDict)
    commentWriter.addDocument(docChunk)
