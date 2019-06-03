import json
import argparse

TEST_PATH = "../../data4Modeling/SecPassageClassification/testDockets_2019-0430.jsonl"

parser = argparse.ArgumentParser(description='Fetch error examples')

parser.add_argument('--sectionNum',
                    required=True, type=int,
                    help='Number of the section where sentence occurs')
parser.add_argument('--sentenceNum',
                    required=True, type=int,
                    help='Number of the sentence')

args = parser.parse_args()

with open(TEST_PATH) as f:
  i = 0
  for line in f:
    obj = json.loads(line)

    if i == args.sectionNum:
      print("@@@@@@@@@",obj["section_name"])
      print("@@@@@@@@@",obj["section_label"])
      if args.sentenceNum>0:
        sent = obj["sent_text"][args.sentenceNum]
      else:
        sent = []
        for s in obj["sent_text"]:
          sent.append(s.split("[SEP]")[0])
        sent = ' \n'.join(sent)
      print(sent)

    i = i + 1