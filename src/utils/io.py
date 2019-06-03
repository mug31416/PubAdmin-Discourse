import json

def saveData(data, fileName, openFlag):
  """Save data to a file.
    
    @param  data      data: string or bytes
    @param  fileName  a name of the output file
    @param  openFlag  a flag to open files, e.g., 'w', 'wb', 'a'
  """
  with open(fileName, openFlag) as outFile:
    outFile.write(data)

def saveJson(jsonObj, fileName):
  """Write jsonObj to JSON file.

    @param  jsonObj   parsed JSON
    @param  fileName  the name of the output file
  """
  with open(fileName, 'w') as outFile:
    json.dump(jsonObj, outFile)
    
def readJson(fileName):
  """Read and parse JSON from file.

    @param fileName   a file to read

    @return a parsed JSON object.
  """
  with open(fileName, 'r') as inpFile:
    return json.loads(inpFile.read())

READ_DOC_ID_FILE_NAME = 'readDocIds.txt'
READ_COMMENT_ID_FILE_NAME = 'readCommentIds.txt'

import os

def writeIds(fileName, ids):
  with open(fileName, 'w') as outFile:
    for oneId in ids:
      outFile.write(oneId + '\n')

def readIds(fileName, checkExists=True):
  res = set()

  if os.path.exists(fileName):
    with open(fileName, 'r') as inpFile:
      for docId in inpFile:
        docId = docId.strip()
        if docId != '':
          res.add(docId)
  else:
    if checkExists:
      raise Exception('File does not exists: %s' % fileName)

  return res

def readIdsIfExists(fileName):
  return readIds(fileName, checkExists=False)
