from fuzzywuzzy import fuzz

def _getTPFPFN(predAuthors, trueAuthors):
    TP, FP, FN = 0, 0, 0 
    predLength = len(predAuthors)
    trueLength = len(trueAuthors)
    if trueLength != predLength:
        raise Exception("size not compatible")
    cpyPredAuthors = predAuthors
    cpyTrueAuthors = trueAuthors
    for index in range(0, predLength):
        predCandidates = cpyPredAuthors[index]
        trueCandidates = cpyTrueAuthors[index]
        
        calculateFN = len(trueCandidates)
        calculateFP = len(predCandidates)
        calculateTP = 0
        for predCandidate in predCandidates:
            res = True in (fuzz.ratio(predCandidate, trueCandidate) > 80 for trueCandidate in trueCandidates)
            if res:
                calculateTP = calculateTP + 1
        FN = FN + calculateFN - calculateTP
        FP = FP + calculateFP - calculateTP
        TP = TP + calculateTP
    
    return TP, FP, FN

def _getPrecision(TP, FP):
    return TP / (TP + FP)

def _getRecall(TP, FN):
    return TP / (TP + FN)

def _getF1Score(precision, recall):
    return (2*precision*recall)/(precision+recall)

def getEvaluation(predAuthors, trueAuthors, verbose):
    TP, FP, FN = _getTPFPFN(predAuthors, trueAuthors)
    
    precision = _getPrecision(TP, FP)
    recall = _getRecall(TP, FN)
    f1Score = _getF1Score(precision, recall)


    if verbose > 0:
        print("TP: {0}, FP: {1}, FN: {2}".format(TP,FP,FN))
        print("precision is {0}".format(precision))
        print("recall is {0}".format(recall))
        print("f1Score is {0}".format(f1Score))
    
    return precision, recall, f1Score



