import numpy as np
import json



fileName1 = "data4Modeling/SecTitleClassification/sectitle_dev.jsonl"
fileName2 = "data4Modeling/SecTitleClassification/sectitle_train.jsonl"

docket1= []
document1 =[]
docket2= []
document2 =[]

with open(fileName1) as f:
  for line in f:
    obj = json.loads(line)
    keys = obj["meta"]
    for key in keys:
      if key.startswith("EPA"):
        docket = ' '.join(key.split("-")[0:4])
        docket1.append(docket)
        document1.append(key)

with open(fileName2) as f:
  for line in f:
    obj = json.loads(line)
    keys = obj["meta"]
    for key in keys:
      if key.startswith("EPA"):
        docket = ' '.join(key.split("-")[0:4])
        docket2.append(docket)
        document2.append(key)


print(len(set(docket1)))
print(len(set(document1)))
print(len(set(docket2)))
print(len(set(document2)))
