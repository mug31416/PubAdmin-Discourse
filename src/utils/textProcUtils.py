"""
Task 1: Entity recognition
utilities to process comment titles
"""
from copy import copy

#DEFAULT_REM_TOKENS = [',', '.', '(', ')', '\r', '\n','\t',':', '\'','\"']

# We actually do want to keep round brackets
DEFAULT_REM_TOKENS = [',', '.', ':', '?', '!', '\r', '\n','\t']

def compressStr(instr, remTokens=DEFAULT_REM_TOKENS, lower=False):
    if instr == '':
        return ('', [])

    remList = [' ']
    if remTokens is not None:
        remList.extend(remTokens)

    condidx = []
    strLen = len(instr)
    for i in range(strLen):
        if instr[i] in remList:
            continue
        else:
            condidx.append(i)

    # No good characters
    if not condidx:
        return ('', [])

    if lower is True:
        outstr = copy(instr[condidx[0]].lower())
    else:
        outstr = copy(instr[condidx[0]])

    idxwalk = [condidx[0]]

    for j in range(1,len(condidx),1):
        if lower is True:
            char = copy(instr[condidx[j]].lower())
        else:
            char = copy(instr[condidx[j]])

        if condidx[j-1] + 1 == condidx[j]:
            outstr = outstr + char
            idxwalk.extend([condidx[j]])

        else:
            outstr = outstr + ' ' + char
            idxwalk.extend([-1, condidx[j]])

    return outstr, idxwalk

import csv

def readerCSV(fileName, delim=',', qchar='"'):
    with open(fileName) as fn:
        reader = csv.reader(fn, delimiter = delim, quotechar = qchar)

        posToName = []

        ln = 1
        for row in reader:
            if ln == 1:
                header = row
                for i in range(0, len(header)):
                    posToName.append(header[i])
            else:
                res = dict()
                if (len(row) != len(header)):
                    raise Exception('Field # mismatch in line %d, header qty: %d, line qty %d:' %
                                    (ln, len(header), len(row)))
                for i in range(0, len(row)):
                    res[posToName[i]] = row[i]

                yield res

            ln += 1

def frsOrgReader(fileName, fieldName):
    for row in readerCSV(fileName):
        yield row[fieldName]

def yagoOrgReader(fileName):
    with open(fileName) as f:
        for row in f:
            yield row.split('\t')[0]

def removeBrackets(orgName):
    start = orgName.find('(')
    if start >= 0:
        end = orgName.rfind(')')
        if start <= end:
            orgName = orgName[:start] + orgName[end+1:]
    return orgName

def plainFileReader(fileName):
    for row in open(fileName):
        row = row.strip()
        if row != '':
            yield row