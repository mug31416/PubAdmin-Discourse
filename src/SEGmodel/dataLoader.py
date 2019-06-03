#!/usr/bin/env python3
'''
Data loader for training
- loads data from JSONL
- optionally prepares multiclass or single class annotations
- creates instances from a section where sentence is an element of the
  sequence
- each sentence is represented by an embedding
'''
import numpy as np
import json

MAX_Y = "ALL"

LABEL_DICT = {"COMMENT_DISCUSSION": [],
             "CHANGE_IN_RULE": [],
             "NO_CHANGE_IN_RULE": [],
             MAX_Y: []}

# Obtain the list of stop words
# from the file and expand it with some
# additional keywords
def getStopWordSet(fileName):
  lst = ["n't", "'s", "'m", "'d", "'ve"]
  with open(fileName, 'r', encoding='utf-8') as inFile:
    for line in inFile:
      lst.append(line.strip())
  return set(lst)


def loadJSONL(fileName):

  listOfSentLists = []
  listOfHeaderTextLists = []
  listOfHeaderFlagLists = []


  labelDict = {}
  for l in LABEL_DICT:
    labelDict[l]=[]

  with open(fileName) as f:
    for line in f:
      obj = json.loads(line)

      sentList = obj["sent_text"]
      listOfSentLists.append(sentList)
      sentQty = len(sentList)

      headerTxt = obj["section_name"]
      headerFlag = obj["section_label"]

      headerTxtList = []
      headerFlagList = []

      for s in range(sentQty):
        headerTxtList.append(headerTxt)
        headerFlagList.append(headerFlag)

      listOfHeaderTextLists.append(headerTxtList)
      listOfHeaderFlagLists.append(headerFlagList)

      labMax = np.zeros(sentQty)

      for labelType in LABEL_DICT:
        if labelType != MAX_Y:
          labs = np.array(obj["sent_labels"][labelType])
          assert(sentQty == len(labs))
          labelDict[labelType].append( (labs != 0).astype(int))
          labMax = np.maximum(labMax, labs)

      labelDict[MAX_Y].append( (labMax != 0).astype(int))

      #print(labelDict)

  return listOfSentLists, listOfHeaderTextLists, listOfHeaderFlagLists, labelDict