import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from nltk.corpus import stopwords


def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)


def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def constructLabeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences


def clean_data(path):
    # vector_dimension=300
    # path="Temp.csv"
    data = pd.read_csv(path)
    # missing_rows = []
    # for i in range(len(data)):
    #     if data.loc[i, 'REVIEW_TEXT'] != data.loc[i, 'REVIEW_TEXT']:
    #         missing_rows.append(i)
    # data = data.drop(missing_rows).reset_index().drop(['index','DOC_ID'],axis=1)
    for i in range(len(data)):
        data.loc[i, 'REVIEW_TEXT'] = cleanup(data.loc[i,'REVIEW_TEXT'])
        data.loc[i, 'REVIEW_TITLE'] = cleanup(data.loc[i,'REVIEW_TITLE'])
        print(i)
    data.loc[data["VERIFIED_PURCHASE"] == "Y", "VERIFIED_PURCHASE"] = 1
    data.loc[data["VERIFIED_PURCHASE"] == "N", "VERIFIED_PURCHASE"] = 0
    data.loc[data["RATING"] == 0, "RATING"] = 0
    data.loc[data["RATING"] == 1, "RATING"] = 1
    data.loc[data["RATING"] == 2, "RATING"] = 2
    data.loc[data["RATING"] == 3, "RATING"] = 3
    data.loc[data["RATING"] == 4, "RATING"] = 4
    data.loc[data["RATING"] == 5, "RATING"] = 5
    #Shuffle Data
    # data = data.sample(frac=1).reset_index(drop=True)
    return data


def getEmbeddings_text(data,vector_dimension=300):

    print("Start")
    x_text = constructLabeledSentences(data['REVIEW_TEXT'])
    y = data['LABEL'].values
    rating = data['RATING'].values
    vp = data['VERIFIED_PURCHASE'].values

    print("Model....")

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10, seed=1)
    text_model.build_vocab(x_text)
    text_model.train(x_text[:21000], total_examples=text_model.corpus_count, epochs=text_model.epochs)

    print("Array....")

    train_size = 16800
    test_size = 4200

    pred_size=len(y)-21000
    # train_size = int(0.8 * len(y))
    # test_size = len(y) - train_size

    # vp_pred_arrays,rating_pred_arrays,xtr_r,xte_r,xtr_vp,xte_vp

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    xtr_r = np.zeros(train_size)
    xte_r = np.zeros(test_size)
    xtr_vp = np.zeros(train_size)
    xte_vp = np.zeros(test_size)

    text_pred_arrays = np.zeros((pred_size, vector_dimension))
    vp_pred_arrays = np.zeros(pred_size)
    rating_pred_arrays = np.zeros(pred_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]
        xtr_r[i] = rating[i]
        xtr_vp[i] = vp[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        xte_r[j] = rating[i]
        xte_vp[j] = vp[i]
        j = j + 1

    j = 0
    for i in range(train_size + test_size, train_size + test_size + pred_size):
        text_pred_arrays[j] = text_model.docvecs['Text_' + str(i)]
        rating_pred_arrays[j] = rating[i]
        vp_pred_arrays[j] = vp[i]
        j = j + 1

    return vp_pred_arrays,rating_pred_arrays,xtr_r,xte_r,xtr_vp,xte_vp,text_train_arrays, text_test_arrays, train_labels, test_labels,text_pred_arrays


def getEmbeddings_title(data,vector_dimension=50):
    
    print("Start")

    x_text = constructLabeledSentences(data['REVIEW_TITLE'])
    y = data['LABEL'].values

    print("Model....")

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,
                            seed=1)
    text_model.build_vocab(x_text)
    text_model.train(x_text[:21000], total_examples=text_model.corpus_count, epochs=text_model.epochs)


    print("Array....")

    train_size = 16800
    test_size = 4200

    pred_size=len(y)-21000
    # train_size = int(0.8 * len(y))
    # test_size = len(y) - train_size

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))

    
    text_pred_arrays = np.zeros((pred_size, vector_dimension))

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        j = j + 1

        
    j = 0
    for i in range(train_size + test_size, train_size + test_size + pred_size):
        text_pred_arrays[j] = text_model.docvecs['Text_' + str(i)]
        j = j + 1

    return text_train_arrays, text_test_arrays,text_pred_arrays
