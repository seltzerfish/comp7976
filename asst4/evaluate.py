import Data_Utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
from scipy.spatial import distance

def fit_classifier(classifier, X, Y, mask):
    train_labels = Y
    # tf-idf
    newx = []
    for sample in X:
        news = []
        for i, e in enumerate(sample):
            if mask[i]:
                news.append(e)
        newx.append(news)
    CU_train_data = np.array(newx)

    # scaler = StandardScaler()

    # tfidf = TfidfTransformer(norm=None)
    # dense = Data_Utils.DenseTransformer()
    # tfidf.fit(CU_train_data)
    # CU_train_data = dense.transform(tfidf.transform(CU_train_data))
    # scaler.fit(CU_train_data)
    # CU_train_data = scaler.transform(CU_train_data)
    # # normalization
    # CU_train_data = normalize(CU_train_data)
    train_data =  CU_train_data
    # evaluation

    classifier.fit(train_data, train_labels)

def get_diff(classifier, goal, sample):
    goal = np.array(goal)
    tfidf = TfidfTransformer(norm=None)
    samp = np.array(sample).reshape(1, -1)
    # dense = Data_Utils.DenseTransformer()
    # tfidf.fit(samp)
    # samp = dense.transform(tfidf.transform(samp))
    dec = classifier.decision_function(samp)
    # print(classifier.predict(np.array(sample).reshape(1, -1)))
    norm1 = dec / np.linalg.norm(dec)
    dist = distance.euclidean(goal, norm1)
    # print(classifier.predict(samp))
    return dist


def get_accuracy(classifier, X, Y):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    fold_accuracy = []
    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = Data_Utils.DenseTransformer()

    for train, test in skf.split(X, Y):
        #train split
        CU_train_data = X[train]
        train_labels = Y[train]
        
        #test split
        CU_eval_data = X[test]
        eval_labels = Y[test]

        # tf-idf
        tfidf.fit(CU_train_data)
        CU_train_data = dense.transform(tfidf.transform(CU_train_data))
        CU_eval_data = dense.transform(tfidf.transform(CU_eval_data))
        
        # standardization
        scaler.fit(CU_train_data)
        CU_train_data = scaler.transform(CU_train_data)
        CU_eval_data = scaler.transform(CU_eval_data)

        # normalization
        CU_train_data = normalize(CU_train_data)
        CU_eval_data = normalize(CU_eval_data)

        train_data =  CU_train_data
        eval_data = CU_eval_data
        # evaluation
        classifier.fit(train_data, train_labels)
        classifier_acc = classifier.score(eval_data, eval_labels)
        # print(classifier.decision_function(eval_data[0:1]), eval_labels)
        fold_accuracy.append((classifier_acc,))

    return np.mean(fold_accuracy, axis = 0)
