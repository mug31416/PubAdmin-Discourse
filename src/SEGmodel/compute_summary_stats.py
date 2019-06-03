#!/usr/bin/env python3
'''
Data loader for training
- loads data from JSONL
- optionally prepares multiclass or single class annotations
- creates instances from a section where sentence is an element of the
  sequence
- each sentence is represented by an embedding
'''
import sys
import argparse
import numpy as np
import json


MAX_Y = "MAX"

def flatten(sentList):

  res = []

  for sent in sentList:
    res.extend(sent.split())

  return res

def wordWeights(sentList):

  res = []

  for sent in sentList:
    res.append(len(sent.split(' ')))

  #print(res)
  return res

def procJSONL(fileName):


  res_dict = dict()
  res_dict['Dockets'] = set()
  res_dict['Documents'] = set()

  res_dict['Sections'] = {  "ALL": 0,
                            "COMMENT_DISCUSSION": 0,
                            "CHANGE_IN_RULE": 0,
                            "NO_CHANGE_IN_RULE": 0,
                            MAX_Y : 0}

  res_dict['Sentences'] = { "ALL": 0,
                            "COMMENT_DISCUSSION": 0,
                            "CHANGE_IN_RULE": 0,
                            "NO_CHANGE_IN_RULE": 0,
                            MAX_Y : 0}

  res_dict['Words'] = {   "ALL": 0,
                          "COMMENT_DISCUSSION": 0,
                          "CHANGE_IN_RULE": 0,
                          "NO_CHANGE_IN_RULE": 0,
                          MAX_Y : 0}

  with open(fileName) as f:
    for line in f:
      obj = json.loads(line)

      res_dict['Dockets'].add(obj["docketId"])
      res_dict['Documents'].add(obj["doc_id"])

      res_dict['Sections']['ALL'] = res_dict['Sections']['ALL'] + 1

      sentList = obj["sent_text"]
      wordList = flatten(sentList)
      wordWeightList = wordWeights(sentList)


      res_dict['Sentences']['ALL'] = res_dict['Sentences']['ALL'] + len(sentList)
      res_dict['Words']['ALL'] = res_dict['Words']['ALL'] + len(wordList)

      labMax = np.zeros(len(sentList),dtype=int)

      for labelType in ['COMMENT_DISCUSSION','CHANGE_IN_RULE',"NO_CHANGE_IN_RULE"]:

        labs = obj["sent_labels"][labelType]
        assert(len(sentList) == len(labs))

        res_dict['Words'][labelType] = res_dict['Words'][labelType] + sum(np.array(labs) * np.array(wordWeightList))
        res_dict['Sentences'][labelType] = res_dict['Sentences'][labelType] + sum(labs)
        res_dict['Sections'][labelType] = res_dict['Sections'][labelType] + np.max(labs)

        labMax = np.maximum(labMax, labs)

      res_dict['Words'][MAX_Y] = res_dict['Words'][MAX_Y] + sum(np.array(labMax) * np.array(wordWeightList))
      res_dict['Sentences'][MAX_Y] = res_dict['Sentences'][MAX_Y] + sum(labMax)
      res_dict['Sections'][MAX_Y] = res_dict['Sections'][MAX_Y] + np.max(labMax)

  return res_dict

def main(argList):

  parser = argparse.ArgumentParser(description='Compute summary statistics')

  parser.add_argument('--file',  type=str, required=True,
                      metavar = 'training file', help='An input JSONL file for processing')

  args = parser.parse_args(argList)

  res = procJSONL(args.file)

  print("Dockets    :", len(res["Dockets"]))
  print("Documents    :", len(res["Documents"]))
  print("Sections:")
  print(res["Sections"])
  print("Sentences:")
  print(res["Sentences"])
  print("Words:")
  print(res["Words"])

if __name__ == '__main__':
  main(sys.argv[1:])