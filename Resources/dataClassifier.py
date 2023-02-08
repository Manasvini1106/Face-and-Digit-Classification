# This file contains feature extraction methods and harness 
# code for data classification

import mostFrequent
import naiveBayes
import customClassifier 
import samples
import sys
import util

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## Describe your enhanced features here...
  
  ##
  """
  features =  basicFeatureExtractorDigit(datum)

  # Your code here to improve features!
  
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features =  basicFeatureExtractorFace(datum)

  # (Optional) Your code here to improve features!

  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print "==================================="
          print "Mistake on example %d" % i 
          print "Predicted %d; truth is %d" % (prediction, truth)
          print "Image: "
          print rawTestData[i]
          break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print "new features:", pix
            continue
      print image  

def readCommand( argv ):
  """
  Processes the command used to run from the command line.
  """
  import getopt

  # Set default options
  options = {'classifier': 'mostfrequent', 
             'data': 'digits', 
             'enhancedFeatures': False,
             'train': 100,
             'odds': False,
             'class1': 1,
             'class2': 0,
             'smoothing': 1,
             'automaticSmooth' : False}
             
  args = {} # This dictionary will hold the objects used by the main method
  
  # Read input from the command line
  commands = ['help', 
              'classifer=', 
              'data=',
              'train=', 
              'enhancedFeatures', 
              'odds',
              'class1=',
              'class2=',
              'smoothing=',
              'automaticSmooth']
  try:
    opts = getopt.getopt( argv, "hc:d:t:fo1:2:k:ai:", commands )
  except getopt.GetoptError:
    print USAGE_STRING
    sys.exit( 2 )
    
  for option, value in opts[0]:
    if option in ['--help', '-h']:
      print USAGE_STRING
      sys.exit( 0 )
    if option in ['--classifier', '-c']:
      options['classifier'] = value
    if option in ['--data', '-d']:
      options['data'] = value
    if option in ['--train', '-t']:
      options['train'] = int(value)
    if option in ['--enhancedFeatures', '-f']:
      options['enhancedFeatures'] = True
    if option in ['--odds', '-o']:
      options['odds'] = True
    if option in ['--class1', '-1']:
      options['class1'] = int(value)
    if option in ['--class2', '-2']:
      options['class2'] = int(value)
    if option in ['--smoothing', '-k']:
      options['smoothing'] = float( value )
    if option in ['--automaticSmooth', '-a']:
      options['automaticSmooth'] = True
    
  # Set up variables according to the command line input.
  print "Doing classification"
  print "--------------------"
  print "data:\t\t" + options['data']
  print "classifier:\t\t" + options['classifier']
  print "using enhanced features?:\t" + str(options['enhancedFeatures'])
  print "training set size:\t" + str(options['train'])
  if(options['data']=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options['enhancedFeatures']):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
  elif(options['data']=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options['enhancedFeatures']):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace      
  else:
    print "Unknown dataset", options['data']
    print USAGE_STRING
    sys.exit(2)
    
  if(options['data']=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)
    
  if options['train'] <= 0:
    print "Training set size should be a positive integer (you provided: %d)" % options['train']
    print USAGE_STRING
    sys.exit(2)
    
  if options['smoothing'] <= 0:
    print "Please provide a positive number for smoothing (you provided: %f)" % options['smoothing']
    print USAGE_STRING
    sys.exit(2)
    
  if options['odds']:
    for className in ['class1','class2']:
      if options[className] not in legalLabels:
        print "Didn't provide a legal labels for the odds ratio for %s" % className
        print USAGE_STRING
        sys.exit(2)

  if(options['classifier'] == "mostfrequent"):
    classifier = mostFrequent.MostFrequentClassifier(legalLabels)
  elif(options['classifier'] == "custom"):
    classifier = customClassifier.CustomClassifier(legalLabels)
  elif(options['classifier'] == "naivebayes"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options['smoothing'])
    if (options['automaticSmooth']):
        print "using automatic smoothing for naivebayes"
        classifier.automaticTuning = True
    else:
        print "using smoothing parameter k=%f for naivebayes" %  options['smoothing']
  else:
    print "Unknown classifier:", options['classifier']
    print USAGE_STRING
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naivebayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with class1=3 vs. class2=6
  
  OPTIONS:    --help, -h
                  display this help
              --classifer, -c
                  chooses the classifier
                  legal values: mostfrequent, naivebayes, custom 
                  default: mostfrequent
              --data, -d
                  chooses the type of dataset
                  legal values: digits, faces
                  default: digits
              --train, -t
                  chooses the size of the training dataset
                  legal values: a positive integer
                  default: 100
              --enhancedFeatures, -f
                  uses your enhanced features instead of just the basic
                  default: False
              --odds, -o
                  selects whether to compute and display the odds ratio analysis
                  default: False
              --class1, -1
                  chooses which class1 to use in the odds ratio analysis
                  legal values: 0,1 for faces; 0,1,..., 9 for digits
                  default: 1
              --class2, -2
                  chooses which class2 to use in the odds ratio analyiss
                  legal values: same as --class1
                  default: 0
              --smoothing, -k
                  set the smoothing parameters for Naive Bayes
                  (if automaticSmooth is on, this will have no effect)
                  legal values: positive real number
                  default: 1
              --automaticSmooth, -a
                  used to activate the automatic smoothing in your classifier
                  default: False
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
      
  # Load data  
  numTraining = options['train']

  if(options['data']=="faces"):
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", TEST_SET_SIZE,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", TEST_SET_SIZE)
    rawTestData = samples.loadDataFile("facedata/facedatatest", TEST_SET_SIZE,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", TEST_SET_SIZE)
  else:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", TEST_SET_SIZE)
    rawTestData = samples.loadDataFile("digitdata/testimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", TEST_SET_SIZE)
    
  
  # Extract features
  print "Extracting features..."
  trainingData = map(featureFunction, rawTrainingData)
  validationData = map(featureFunction, rawValidationData)
  testData = map(featureFunction, rawTestData)
  
  # Conduct training and testing
  print "Training..."
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  print "Validating..."
  guesses = classifier.classify(validationData)
  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
  print "Testing..."
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
  util.pause()
  analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  
  # do odds ratio computation if specified at command line
  if((options['odds']) & (options['classifier'] != "mostfrequent")):
    class1, class2 = options['class1'], options['class2']
    features_class1,features_class2,features_odds = classifier.findHighOddsFeatures(class1,class2)
    if(options['classifier'] == "naivebayes"):
      string1 = "=== Features with max P(F_i = on | class = %d) ===" % class1
      string2 = "=== Features with max P(F_i = on | class = %d) ===" % class2
      string3 = "=== Features with highest odd ratio of class %d over class %d ===" % (class1, class2)
    else:
      string1 = "=== Features with largest weight for class %d ===" % class1
      string2 = "=== Features with largest weight for class %d ===" % class2
      string3 = "=== Features with for which weight(class %d)-weight(class %d) is biggest ===" % (class1, class2)    
      
    print string1
    printImage(features_class1)
    print string2
    printImage(features_class2)
    print string3
    printImage(features_odds)

if __name__ == '__main__':
  # Read input
  args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier(args, options)
