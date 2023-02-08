import util
import classificationMethod

class CustomClassifier(classificationMethod.ClassificationMethod):
  """
  Your classifier.  Add a description here.
  """
  def __init__(self, legalLabels):
    # You may pass additional arguments to this function if neeed.
    self.guess = None
    self.type = "custom"
    # You may add additional initialization code here.
  
  def train(self, data, labels, validationData, validationLabels):
    # You will need to replace the implementation below with your own.
    # This implementation is identical to the MostFrequenceClassifier.
    """
    Find the most common label in the training data.
    """
    counter = util.Counter()
    counter.incrementAll(labels, 1)
    self.guess = counter.argMax()
  
  def classify(self, testData):
    # You will need to replace the implementation below with your own.
    # This implementation is identical to the MostFrequenceClassifier.
    """
    Classify all test data as the most common label.
    """
    return [self.guess for i in testData]

  # You may include additional methods here if needed.
