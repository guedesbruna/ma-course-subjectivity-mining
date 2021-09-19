import pandas as pd

class result:
  def __init__(self, predictions=None,classification_report=None):
    if predictions is None:
        self.predictions = pd.DataFrame({'A' : []})
        self.classification_report = {}
    else:
        self.predictions = predictions
        self.classification_report = classification_report

def save_all_predictions(test_X, test_y, sys_y):
    #cols: predictions, golden labels, tweet
    return pd.DataFrame(zip(sys_y,test_y.values,test_X),columns=['pred','gold','tweet'])

