#!/usr/bin/env python
import sys, os
sys.path.append('.')

import spacy

from utils.io import *

from utils.misc import *
from utils.spacyAnnot import *
from utils.parse import *

if len(sys.argv) != 4:
  print('Usage <docket dir> <list of exlcuded dockets> <output file>')
  sys.exit(1)

docketDir = sys.argv[1]
exclDocketIds = readIds(sys.argv[2])
outFile = sys.argv[3]

startYear = 0
docNum = 0

fOut = open(outFile, 'w')

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'pos'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))


for docketId, docId, postedDate, isFinal, ext, htmlFileName \
  in getDocInfoFromMetaFull(docketDir, startYear):

  if docketId in exclDocketIds:
    print(f'Exclude {docId} because it is on the list of excluded dockets!')
    continue

  if htmlFileName is None or ext != 'html':
    print(f'Ignoring {docId} because it has no HTML file')
    continue

  if not os.path.exists(htmlFileName):
    print('Ignoring missing file:', htmlFileName)
    continue

  print('Processing: ', htmlFileName)

  docNum += 1
  sectNums, sectNames, sectParas = partialParse(htmlFileName)

  for i in range(len(sectNames)):
    # sectParas is a list of list of paragraphs

    sectTitle = sectNames[i]
    fOut.write(sectTitle + '\n')

    res = list(nlp.pipe(sectParas[i]))
    assert(len(res) == len(sectParas[i]))
    for parId in range(len(res)):
      oneParaText = sectParas[i][parId]
      oneParaDoc = res[parId]
      for oneSent in oneParaDoc.sents:
        start = oneSent.start_char
        end = oneSent.end_char
        fOut.write(oneParaText[start:end] + '\n')

    fOut.write('\n')  # Document/section separator

  print('Processed %d documents' % docNum)


fOut.close()