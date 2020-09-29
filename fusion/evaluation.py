from sklearn.metrics import precision_score, recall_score, f1_score

## Using an array of 1 and 0. Where y_true is all 1, 
## if the prediction is correct set to 1

def getPrecision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='binary', zero_division=1)

def getRecall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='binary', zero_division=1)

def getF1Score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='binary', zero_division=1)


