import spacy, os, tempfile

#https://github.com/sina-al/pynlp - how to install and run pynlp
from pynlp import StanfordCoreNLP

from whoosh.index import open_dir
from whoosh.query import Or, And, Term

import _pickle as cPickle

from ahocorasick import Automaton


STOP_WORDS = frozenset(('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can',
                        'for', 'from', 'have', 'if', 'in', 'is', 'it', 'may',
                        'not', 'of', 'on', 'or', 'tbd', 'that', 'the', 'this',
                        'to', 'us', 'we', 'when', 'will', 'with', 'yet',
                        'you', 'your', 'inc'))

from utils.textProcUtils import  compressStr

class Annotation:
    def __init__(self, startChar, endChar, annotType, annotSubType = None):
        self.startChar = startChar
        self.endChar = endChar
        self.annotType = annotType
        self.annotSubType = annotSubType
    def __repr__(self):
        return '(%d, %d) %s%s' % \
            (self.startChar, self.endChar, self.annotType, ''
            if self.annotSubType is None else '/' + self.annotSubType)

def computeBestOverlap(q, annotSet):
    '''
    Compute the best overlap between the annotation q and a set of annotations.

    :param q: a query annotation
    :param annotSet: a set of annotations to search
    :return: the best overlap score.
    '''
    best = 0

    qs, qe = q
    for an in annotSet:
        ds, de = an
        # [ds de]
        #   [qs  qe]
        #     [ds de]
        overlap = max(0, min(qe, de) - max(qs, ds))/max(qe-qs,de-ds)
        if overlap > best:
            best = overlap

    return best

def computeFuzzyIntersection(set1, set2):
    res = 0

    for an in set1:
        res += computeBestOverlap(an, set2)

    return res

class Annotator:
    def getAnnotations(self, text):
        return []
    def cleanUp(self):
        pass



class SpacyAnnotator(Annotator):
    def __init__(self):
        self.nlp = spacy.load('en')

    def getAnnotations(self, commentText):
        resAnnot = []

        if commentText == '':
            return resAnnot

        doc = self.nlp(commentText)

        for e in doc.ents:
            if e.label_ in ['ORG', 'NORP', 'FACILITY']:
                resAnnot.append(Annotation(e.start_char, e.end_char, 'spacy', e.label_))

        return resAnnot

class AhoAnnotator(Annotator):
    def __init__(self, indexDir, minWordQty):
        self.automatons = []
        self.minWordQty = minWordQty
        for fn in os.listdir(indexDir):
            self.automatons.append(cPickle.load(open(os.path.join(indexDir, fn), 'rb')))

    def getAnnotations(self, text):
        rawResAnnot = []

        procCommText, origIndx = compressStr(text, lower=True)

        seen = set()
        textLen = len(text)

        for automaton in self.automatons:
            for endIndex, (_, matchLen) in automaton.iter(procCommText):
                if endIndex + 1 < len(procCommText) and procCommText[endIndex + 1] != ' ' :
                    continue # Not a word boundary
                startIndex = endIndex - matchLen + 1
                if startIndex > 0 and procCommText[startIndex - 1] != ' ':
                    continue # Not a word boundary

                startChar = origIndx[startIndex]
                endChar = origIndx[endIndex] + 1

                # Let's check for duplicates
                indxPair = (startChar, endChar)
                if indxPair in seen:
                    continue
                seen.add(indxPair)

                annotText= text[startChar:endChar]
                flag=text[startChar].lower() != text[startChar]
                qty = len(annotText.split())
                if flag and qty >= self.minWordQty:

                    if endChar < textLen and endChar > startChar and text[endChar-1] != '.' and text[endChar] == '.':
                        endChar += 1

                    rawResAnnot.append(Annotation(startChar, endChar, 'OrgDict'))

        resAnnot = []

        for i in range(len(rawResAnnot)):
            flag=True
            an1=rawResAnnot[i]
            for k in range(len(rawResAnnot)):
                if i == k: continue
                an2 = rawResAnnot[k]
                if an1.startChar >= an2.startChar and an1.endChar <= an2.endChar:
                    flag=False
                    break
            if flag:
                resAnnot.append(an1)

        return resAnnot

class StanfordAnnotator(Annotator):
    def __init__(self):
        annotators = 'tokenize, ssplit, pos, lemma, ner'
        options = {'openie.resolve_coref': False}

        self.nlp = StanfordCoreNLP(annotators=annotators, options=options)

    def getAnnotations(self, commentText):
        resAnnot = []

        if commentText == '':
            return resAnnot

        doc = self.nlp(commentText.encode(encoding='utf-8'))

        for ent in doc.entities:
            if ent.type == 'ORGANIZATION':
                toks = ent._tokens
                resAnnot.append(Annotation(toks[0].beginChar, toks[-1].endChar, 'stanford', ent.type))

        return resAnnot

from ncrfpp.main import evaluate
from ncrfpp.model.seqmodel import SeqModel
import torch
from ncrfpp.utils.data import Data

class NCRFAnnotator(Annotator):
    def __init__(self, dSetFileName, modelFileName):
        """
        Create a NCRF++ constructor.

        :param dSetFileName:    a dset-configuration file name.
        :param modelFileName:   a model file name.
        """

        data = Data()
        data.HP_gpu = torch.cuda.is_available()
        data.nbest = None # TODO try a non-None nbest > 1
        data.load(dSetFileName)
        model = SeqModel(data)
        model.load_state_dict(torch.load(modelFileName))
        self.data = data
        self.model = model
        self.nlp = spacy.load('en')

        f, self.inpFileName = tempfile.mkstemp()
        os.close(f)

        self.data.raw_dir = self.inpFileName

    def cleanUp(self):
        os.unlink(self.inpFileName)

    def getAnnotations(self, commentText):
        resAnnot = []
        if commentText == '':
            return []
        doc = self.nlp(commentText)
        tokStr = []
        tokStart = []
        tokEnd = []
        with open(self.inpFileName, 'w') as f:
            for tok in doc:
                text = tok.text.strip()
                if text == '':
                    continue
                tokStr.append(text)
                tokStart.append(tok.idx)
                tokEnd.append(tok.idx + len(tok.text))
                # Appending a fake tag b/c otherwise
                # the NCRFPP function we call will break
                f.write(tok.text + ' O\n')
            f.write('\n')
        tokQty = len(tokStr)
        self.data.generate_instance('raw')
        speed, acc, p, r, f, predResults, predScores = evaluate(self.data, self.model, 'raw', self.data.nbest)
        if len(predResults) != 1 and len(predResults[0]) != tokQty:
            raise Exception('Bug: # of predicted items does not match # of input items!')
        # We use only one sentence
        predResults = predResults[0]
        start = 0
        while start < tokQty:
            if predResults[start] == 'O':
                start += 1
                continue
            end = start

            while end < tokQty:
                tagPos, tagVal = predResults[end].split('-')
                if tagPos == 'E' or tagPos == 'S' or end + 1 == tokQty:
                    resAnnot.append(Annotation(tokStart[start], tokEnd[end], tagVal))
                    start = end + 1
                    break
                else:
                    end += 1

        return resAnnot


class DictAnnotator(Annotator):
    def __init__(self, indexDir, fieldName, topK = 10, querySpan = 6):
        """
        Create a dictionary annotator.

        :param indexDir:    a directory with the Woosh index
        :param fieldName:   the name of the index field
        :param topK:        the number of top candidate entries in each query
        :param querySpan:   a span of each query in the number of words: we
                            relying on a moving window approach with the window
                            querySpan and string querySpan // 2
        """
        self.searcher = open_dir(indexDir).searcher()
        self.topK = topK
        self.fieldName = fieldName
        self.querySpan = querySpan
        self.queryStride = querySpan // 2

    def getAnnotations(self, commentText):
        tmpRes = {}

        if commentText == '':
            return []

        procCommText, origIndx = compressStr(commentText, lower=True)

        termArr = procCommText.split()

        # There might be repeating entries
        orgNames = set()

        for qs in range(0, len(termArr), self.queryStride):
            qe = min(qs + self.querySpan, len(termArr))

            q = []
            for i in range(qs, qe - 1):
                if not termArr[i] in STOP_WORDS:
                    bigram = And([Term(self.fieldName, termArr[i]), Term(self.fieldName, termArr[i + 1])])
                    q.append(bigram)

            #print('@@', ' '.join(termArr[qs:qe]))
            #print('Query: ', q)

            res = self.searcher.search(Or(q), limit = self.topK)

            #print('Found %d results' % len(res))

            for k in range(len(res)):
                if k >= self.topK:
                    break
                orgName = res[k][self.fieldName]
                orgNames.add(orgName)


        for orgName in orgNames:
            start = 0
            while start < len(procCommText):
                indx = procCommText.find(orgName, start)
                #print('###', orgName, start, indx)
                if indx == -1: break
                assert(indx + len(orgName) <= len(origIndx))
                start = indx + len(orgName)
                # To be a valid match
                startChar = origIndx[indx]
                endChar = origIndx[indx + len(orgName) - 1] + 1
                # TODO additional condtitions for spaces!!
                if  startChar >= 0 and endChar >= 0 :
                    if startChar in tmpRes:
                        tmpRes[startChar] = max(tmpRes[startChar], endChar)
                    else:
                        tmpRes[startChar] = endChar

        resAnnot = []

        for startChar in tmpRes:
            endChar = tmpRes[startChar]
            resAnnot.append(Annotation(startChar, endChar, 'OrgDict'))

        return resAnnot