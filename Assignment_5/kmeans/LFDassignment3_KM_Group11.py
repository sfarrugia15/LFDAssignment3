import itertools
import numpy as np

# Import sklearn models and helpers:
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, v_measure_score, homogeneity_completeness_v_measure

# Import tSNE helpers
from yellowbrick.text import TSNEVisualizer
from sklearn.manifold import TSNE
# from ggplot import *
from sklearn.base import clone


def read_corpus(corpus_file):
    documents = []
    topic_labels = []
    sentiment_labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            # 2-class problem: positive vs negative
            sentiment_labels.append(tokens[1])
            # 6-class problem: books, camera, dvd, health, music, software
            topic_labels.append(tokens[0])

            # if use_sentiment:
            #     # 2-class problem: positive vs negative
            #     sentiment_labels.append(tokens[1])
            # else:
            #     # 6-class problem: books, camera, dvd, health, music, software
            #     topic_labels.append(tokens[0])

    return documents, topic_labels, sentiment_labels


# a dummy function that just returns its input
def identity(x):
    return x


def run_kmeans(Xtrain, Ytrain, Xtest, Ytest, K=6, n_init=1, verbose=1, plotTSNE=False):
    # let's use the TF-IDF vectorizer
    tfidf = True

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor=identity,
                              tokenizer=identity)
    else:
        vec = CountVectorizer(preprocessor=identity,
                              tokenizer=identity)

    ######## RUN K-MEANS ########
    km = KMeans(n_clusters=K, n_init=n_init, verbose=verbose)

    classifier = Pipeline([('vec', vec),
                           ('cls', km)])

    classifier.fit(Xtrain)

    print("\n########## Development scores on train set:")
    print("adjusted rand score: ", adjusted_rand_score(Ytrain, km.labels_))
    print("v measure: ", v_measure_score(Ytrain, km.labels_))

    Yguess = classifier.predict(Xtest)

    print("\n########## Generalization scores on test set:")
    print("adjusted rand score: ", adjusted_rand_score(Ytest, Yguess))
    print("v measure: ", v_measure_score(Ytest, Yguess))

    if plotTSNE:
        # perform_tsne(Xtrain, Ytrain, clusterLabels=True)  # tSNE with gold labels
        perform_tsne(Xtrain, km.labels_, vec=vec, clusterLabels=True)  # tSNE clustering

def test_six_way_classification(Xtrain, Ytrain, Xtest, Ytest):
    run_kmeans(Xtrain, Ytrain, Xtest, Ytest)

    print("\nn_init=5")
    run_kmeans(Xtrain, Ytrain, Xtest, Ytest, n_init=5, verbose=0)

    print("\nn_init=10")
    run_kmeans(Xtrain, Ytrain, Xtest, Ytest, n_init=10, verbose=0)

    print("\nn_init=15")
    run_kmeans(Xtrain, Ytrain, Xtest, Ytest, n_init=15, verbose=0)


def perform_tsne(X, Y, vec=None, outpath="", clusterLabels=False, savePlot=False):
    if vec==None:
        vec = TfidfVectorizer(preprocessor=identity,
                                tokenizer=identity)

    docs = vec.fit_transform(X)
    labels = Y

    # from yellowbrick.text import TSNEVisualizer
    tsne = TSNEVisualizer()

    if clusterLabels:
        tsne.fit(docs, ["c{}".format(c) for c in Y])  # where Y=clusters.labels_
    else:
        tsne.fit(docs, labels)

    if savePlot:
        # tsne.finalize()
        tsne.poof(outpath = outpath)
    else:
        tsne.poof()


def binary_classification(Xtrain, Ytrain, Xtest, Ytest, Ytrain_sentiment, Ytest_sentiment,
                          perform_kmeans=True, plotTSNE=False, class_1="music", class_2="health"):
    class_1 = class_1
    class_2 = class_2

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    Ytrain_sentiment = np.array(Ytrain_sentiment)
    Ytest_sentiment = np.array(Ytest_sentiment)

    Xtrain_two_class = Xtrain[(np.array(Ytrain) == class_1) | (np.array(Ytrain) == class_2)]
    Ytrain_two_class = Ytrain[(np.array(Ytrain) == class_1) | (np.array(Ytrain) == class_2)]
    Xtest_two_class = Xtest[(np.array(Ytest) == class_1) | (np.array(Ytest) == class_2)]
    Ytest_two_class = Ytest[(np.array(Ytest) == class_1) | (np.array(Ytest) == class_2)]

    Ytrain_sentiment_two_class = Ytrain_sentiment[(np.array(Ytrain) == class_1) | (np.array(Ytrain) == class_2)]
    Ytest_sentiment_two_class = Ytest_sentiment[(np.array(Ytest) == class_1) | (np.array(Ytest) == class_2)]

    # print(Xtrain_two_class)
    # print(Ytrain_two_class)
    #
    # print(Xtest_two_class)
    # print(Ytest_two_class)

    if plotTSNE:
        perform_tsne(Xtrain_two_class, Ytrain_two_class, savePlot=False, outpath="output/tsne/topic_class_"+class_1+"_"+class_2+".png")
        perform_tsne(Xtrain_two_class, Ytrain_sentiment_two_class, savePlot=False, outpath="output/tsne/sentiment_class_"+class_1+"_"+class_2+".png")

    if perform_kmeans:
        print("\n##########\nClustering: Topic Labels\n##########")
        run_kmeans(Xtrain_two_class, Ytrain_two_class, Xtest_two_class, Ytest_two_class, K=2)

        print("\n##########\nClustering: Sentiment Labels\n##########")
        run_kmeans(Xtrain_two_class, Ytrain_sentiment_two_class, Xtest_two_class, Ytest_sentiment_two_class, K=2)



def perform_tsne_all_class_combinations(Xtrain, Ytrain, Xtest, Ytest, Ytrain_sentiment, Ytest_sentiment):
    classes = ['books', 'camera', 'dvd', 'health', 'music', 'software']

    # for L in range(0, len(classes) + 1):
    for subset in itertools.combinations(classes, 2):
        # print(subset[0], " - ", subset[1])
        binary_classification(Xtrain, Ytrain, Xtest, Ytest, Ytrain_sentiment, Ytest_sentiment, class_1=subset[0], class_2=subset[1], perform_kmeans=False, plotTSNE=True)



##########
# Main function
##########
if __name__ == '__main__':
    # load data
    X, Ytopic, Ysentiment = read_corpus('trainset.txt')
    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Ytopic[:split_point]
    Ytrain_sentiment = Ysentiment[:split_point]
    Xtest = X[split_point:]
    Ytest = Ytopic[split_point:]
    Ytest_sentiment = Ysentiment[split_point:]

    # print(Xtrain)
    # print(Ytrain)

    # tfidf = TfidfVectorizer(preprocessor=identity,
    #                           tokenizer=identity)
    # docs = tfidf.fit_transform(X)
    # labels = Ytopic
    #
    # tsne = TSNEVisualizer()
    # tsne.fit(docs, labels)
    # tsne.poof()
    # perform_tsne(Xtest, Ytest, savePlot=True, outpath="output/test.png")
    # perform_tsne(Xtest, Ytest, savePlot=True, outpath="output/test2.png")


    ######## Exercise 3.2.1 ########
    print("###############\nExercise 3.2.1\n###############")
    # test_six_way_classification(Xtrain, Ytrain, Xtest, Ytest)


    ######## Exercise 3.2.2 ########
    print("\n\n###############\nExercise 3.2.2\n###############")

    perform_tsne_all_class_combinations(Xtrain, Ytrain, Xtest, Ytest, Ytrain_sentiment, Ytest_sentiment)

    # binary_classification(Xtrain, Ytrain, Xtest, Ytest, Ytrain_sentiment, Ytest_sentiment, plotTSNE=True)