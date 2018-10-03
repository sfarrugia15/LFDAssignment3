from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import itertools
import numpy as np
from collections import Counter
from matplotlib.colors import Normalize
import time
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
from sklearn import svm
np.random.seed(2018)
import progressbar
from time import sleep
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
def read_corpus(corpus_file, test_file):
    reviews = []
    labels = []
    reviewsTest = []
    labelsTest = []
    lemmatizer = WordNetLemmatizer()
    # open the file of the user passed parameter directory with utf-8 encoding as value f
    with open(corpus_file, encoding='utf-8') as f:
        # Read each line
        print("Started reading and processing training data")
        for line in f:

            tokens = word_tokenize(line)
            tokens = [w.lower() for w in tokens]

            table = str.maketrans('', '', string.punctuation)  # Remove punctuation from each word
            stripped = [w.translate(table) for w in tokens]

            words = [word for word in stripped if word.isalpha()]  # Remove tokens that are not alphabetic

            stop_words = set(stopwords.words('english'))  # Remove stop words
            words = [w for w in words if not w in stop_words]

            # stemmed = [porter.stem(word) for word in words] #Stem words

            lemmatized = [lemmatizer.lemmatize(stem) for stem in words]  # Lemmatize

            reviews.append(lemmatized[3:])
            labels.append(tokens[1])

    with open(test_file, encoding='utf-8') as t:
        print ("Started reading and processing test data")
        for line in t:
            tokens = word_tokenize(line)
            tokens = [w.lower() for w in tokens]

            table = str.maketrans('', '', string.punctuation)  # Remove punctuation from each word
            stripped = [w.translate(table) for w in tokens]

            words = [word for word in stripped if word.isalpha()]  # Remove tokens that are not alphabetic

            stop_words = set(stopwords.words('english'))  # Remove stop words
            words = [w for w in words if not w in stop_words]

            lemmatized = [lemmatizer.lemmatize(stem) for stem in words]  # Lemmatize

            reviewsTest.append(lemmatized[3:])
            labelsTest.append(tokens[1])
    return reviews, labels, reviewsTest, labelsTest

# a dummy function that just returns its input
def identity(x):
    return x

def defaultSettings(X, Y, XTest, YTest):
    #X, Y, XTest, YTest = read_corpus(sys.argv[1], sys.argv[2])

    #classLabels = ['books', 'camera', 'dvd', 'health', 'music', 'software']
    classLabels = ['pos', 'neg']
    print('Training class distributions summary: {}'.format(Counter(Y)))
    print('Test class distributions summary: {}'.format(Counter(YTest)))
    print ("-----------------------------------------------------")
    print("Printing test results for SVC using linear kernel and c = 1.0 \n")
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    vec = TfidfVectorizer(preprocessor= identity, tokenizer = identity)

    # combine the vectorizer with an SVM classifier
    classifier = Pipeline([('vec', vec), ('cls',  svm.SVC(kernel="linear", C=1.0))])

    # Train the classifier passing the training data X along with their corresponding labels Y.
    t0 = time.time()
    classifier.fit(X, Y)
    train_time = time.time() - t0
    print('Train time', train_time)

    # predict labels using the classifier on the test set
    t1 = time.time ()
    YGuess = classifier.predict (XTest)
    test_time = time.time () - t1
    print ('Test time', test_time)

    # print out the accuracy value produced by the classifier -
    # comparing the golden standard with the predicted corresponding value
    print ('Classification accuracy on test: {0}'.format (accuracy_score (YTest, YGuess)))
    print ("Printing Classification Report")
    print (classification_report (YGuess, YTest, target_names=classLabels))

    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype ('float') / cm.sum (axis=1)[:, np.newaxis]
            print ("Normalized confusion matrix")
        else:
            print ('Confusion matrix, without normalization')

        print (cm)

        plt.imshow (cm, interpolation='nearest', cmap=cmap)
        plt.title (title)
        plt.colorbar ()
        tick_marks = np.arange (len (classes))
        plt.xticks (tick_marks, classes, rotation=45)
        plt.yticks (tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max () / 2.
        for i, j in itertools.product (range (cm.shape[0]), range (cm.shape[1])):
            plt.text (j, i, format (cm[i, j], fmt), horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout ()
        plt.ylabel ('True label')
        plt.xlabel ('Predicted label')


    cnf_matrix = confusion_matrix (YTest, YGuess, labels=classLabels)
    print(cnf_matrix)
    plt.figure()
    plot_confusion_matrix (cnf_matrix, classes=classLabels)
    plt.show()


def settingC(X, Y, XTest, YTest):
    bar = progressbar.ProgressBar (maxval=15, widgets=[progressbar.Bar ('=', '[', ']'), ' ', progressbar.Percentage ()])
    bar.start ()

    print("Starting computation of different C values - from 0 to 1 with 0.1 intervals")
    print ("-----------------------------------------------------")
    #X, Y, XTest, YTest = read_corpus(sys.argv[1], sys.argv[2])

    classLabels = ['pos', 'neg']
    print('Training class distributions summary: {}'.format(Counter(Y)))
    print('Test class distributions summary: {}'.format(Counter(YTest)))
    print ("-----------------------------------------------------")
    print ("Printing test results for SVC using linear kernel and c from 0 to 1 with 0.1 intervals \n")

    c = [x * 0.1 for x in range (1, 16)]
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    vec = TfidfVectorizer(preprocessor= identity, tokenizer = identity)
    accuracy = []

    for i in range(15):
        bar.update(i + 1)
        # combine the vectorizer with a Naive Bayes classifier
        classifier = Pipeline([('vec', vec), ('cls',  svm.SVC(kernel="linear", C=c[i]))])

        # Train the classifier passing the training data X along with their corresponding labels Y.
        classifier.fit(X, Y)
        YGuess = classifier.predict (XTest)

        # print out the accuracy value produced by the classifier -
        # comparing the golden standard with the predicted corresponding value
        #print ('Classification accuracy on test: {0}'.format (accuracy_score (YTest, YGuess)))
        #print ("Printing Classification Report")
        #print (classification_report (YGuess, YTest, target_names=classLabels))
        accuracy.append(accuracy_score (YTest, YGuess))
    bar.finish()
    print(accuracy, c)
    plt.figure ()
    plt.plot(c, accuracy)
    plt.title("c parameter vs. Accuracy")
    plt.xlabel("c")
    plt.ylabel("Accuracy")
    plt.show()

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def non_linear_kernel(X, Y, XTest, YTest):
    bar = progressbar.ProgressBar (maxval=10, widgets=[progressbar.Bar ('=', '[', ']'), ' ', progressbar.Percentage ()])
    bar.start ()

    print ("Starting computation of different Gamma values - from 0 to 1 with 0.1 intervals")
    print ("-----------------------------------------------------")

    # classLabels = ['books', 'camera', 'dvd', 'health', 'music', 'software']
    classLabels = ['pos', 'neg']
    print ('Training class distributions summary: {}'.format (Counter (Y)))
    print ('Test class distributions summary: {}'.format (Counter (YTest)))

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    vec = TfidfVectorizer (preprocessor=identity, tokenizer=identity)
    accuracy = []
    classifiers = []

    C_range = [x * 0.1 for x in range (1, 11)]
    gamma_range = [x * 0.1 for x in range (1, 11)]

    param_grid = dict (gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit (n_splits=1, test_size=0.2, random_state=42)
    X = vec.fit_transform(X)
    grid = GridSearchCV (SVC(kernel="rbf"), param_grid=param_grid, cv=cv, verbose=100)
    grid.fit (X, Y)

    C_2d_range = [1e-2, 1, 1e2]
    gamma_2d_range = [1e-1, 1, 1e1]
    print ("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
    # for C in C_2d_range:
    #     bar.update (C + 1)
    #     for gamma in gamma_2d_range:
    #
    #         # combine the vectorizer with a SVC
    #         classifier = Pipeline ([('vec', vec), ('cls', svm.SVC (kernel="rbf", gamma=gamma,C=C))])
    #
    #         # Train the classifier passing the training data X along with their corresponding labels Y.
    #         classifier.fit (X, Y)
    #         classifiers.append ((C, gamma, classifier))
    #         YGuess = classifier.predict (XTest)
    #
    #         # print out the accuracy value produced by the classifier -
    #         # comparing the golden standard with the predicted corresponding value
    #         # print ('Classification accuracy on test: {0}'.format (accuracy_score (YTest, YGuess)))
    #         # print ("Printing Classification Report")
    #         # print (classification_report (YGuess, YTest, target_names=classLabels))
    #         accuracy.append (accuracy_score (YTest, YGuess))
    # bar.finish()
    # # bar.finish ()
    # # plt.figure ()
    # # plt.plot (gamma, accuracy)
    # # plt.title ("c parameter vs. Accuracy")
    # # plt.xlabel ("c")
    # # plt.ylabel ("Accuracy")
    # # plt.show ()
    # plt.figure (figsize=(8, 6))
    # xx, yy = np.meshgrid (np.linspace (-3, 3, 200), np.linspace (-3, 3, 200))
    # for (k, (C, gamma, clf)) in enumerate (classifiers):
    #     # evaluate decision function in a grid
    #     Z = clf.decision_function (np.c_[xx.ravel (), yy.ravel ()])
    #     Z = Z.reshape (xx.shape)
    #
    #     # visualize decision function for these parameters
    #     plt.subplot (len (C_2d_range), len (gamma_2d_range), k + 1)
    #     plt.title ("gamma=10^%d, C=10^%d" % (np.log10 (gamma), np.log10 (C)), size='medium')
    #
    #     # visualize parameter's effect on decision function
    #     plt.pcolormesh (xx, yy, -Z, cmap=plt.cm.RdBu)
    #     plt.scatter (X, c=Y, cmap=plt.cm.RdBu_r, edgecolors='k')
    #     plt.xticks (())
    #     plt.yticks (())
    #     plt.axis ('tight')

    #scores = grid.cv_results_['mean_test_score'].reshape (len (C_range), len (gamma_range))
def test():
    c = [x * 0.1 for x in range (1, 10)]
    i = range(10)
    print(c)
    print(i)

def main():
    X, Y, XTest, YTest = read_corpus (sys.argv[1], sys.argv[2])
    defaultSettings(X, Y, XTest, YTest)
    #settingC(X, Y, XTest, YTest)
    #non_linear_kernel(X, Y, XTest, YTest)
    #test()

if __name__ == "__main__":
    main()