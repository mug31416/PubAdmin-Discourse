from utils.annotators import computeFuzzyIntersection

class Evaluator:
    def __init__(self):
        self.totRecall = 0
        self.totPrec = 0
        self.qtyRecall = 0
        self.qtyPrec = 0

    def procOneItem(self, setMan, setAuto):
        return NotImplemented

    def getPrec(self):
        return self.totPrec / max(1.0, self.qtyPrec)

    def getRecall(self):
        return self.totRecall / max(1.0, self.qtyRecall)

    def getFScore(self):
        precision = self.getPrec()
        recall = self.getRecall()
        return 2 * (precision * recall) / (precision + recall)


class FuzzyEvaluator(Evaluator):
    def __init__(self):
        super(FuzzyEvaluator, self).__init__()

    def procOneItem(self, setMan, setAuto):
        if setMan:
            recall = computeFuzzyIntersection(setMan, setAuto) / float(len(setMan))
            self.totRecall += recall
            self.qtyRecall += 1
            print('Fuzzy recall %g' % recall)
        if setAuto:
            prec = computeFuzzyIntersection(setAuto, setMan) / float(len(setAuto))
            self.totPrec += prec
            print('Fuzzy precision %g' % prec)
            self.qtyPrec += 1


class ExactEvaluator(Evaluator):
    def __init__(self):
        super(ExactEvaluator, self).__init__()

    def procOneItem(self, setMan, setAuto):
        if setMan:
            recall = len(setMan.intersection(setAuto)) / float(len(setMan))
            self.totRecall += recall
            self.qtyRecall += 1
            print('Exact recall %g' % recall)
        if setAuto:
            prec = len(setAuto.intersection(setMan)) / float(len(setAuto))
            self.totPrec += prec
            print('Exact precision %g' % prec)
            self.qtyPrec += 1

