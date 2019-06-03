from utils.parse import NL, removeBasicTags

def addNgram(res, ngram, qty):
  if ngram not in res:
    res[ngram] = 0
  res[ngram] += qty

def stringToNgrams(line, N):
  '''Convert a string to a dictionary of n-grams'''
  res = dict()
  for i in range(0, len(line) - N + 1):
    addNgram(res, line[i:i+N], 1)
  return res

def docToNgrams(doc, N):
  res = dict()
  for line in doc.split(NL):
    for ngram, qty in stringToNgrams(line, N).items():
      addNgram(res, ngram, qty)
  return res

def wghtJaccard(x1, x2):
  qty1 = sum(v for _, v in x1.items())
  qty2 = sum(v for _, v in x2.items())
  cmnKeys = set(x1).intersection(set(x2))
  qtyInter = sum([min(x1[k], x2[k]) for k in cmnKeys])
  return float(qtyInter) / max(qty1, qty2)

def maxMissContigLinesQty(srcDoc, trgDoc,
                          delBasicTags=True):
  linesSrc = srcDoc.split(NL)
  lineSetTrg = set(trgDoc.split(NL))

  prevMiss = -1
  res = 0

  startMiss, endMiss = None, None

  for i in range(len(linesSrc)):
    line = linesSrc[i]
    if delBasicTags:
      line = removeBasicTags(line)
    if line in lineSetTrg:
      prevMiss = i
    else:
      if i - prevMiss > res:
        res = i - prevMiss
        startMiss = prevMiss
        endMiss = i

  return res, startMiss, endMiss