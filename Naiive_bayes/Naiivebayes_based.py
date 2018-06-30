import re
import sys
import os
import sklearn.datasets
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import string
from nltk.stem.snowball import EnglishStemmer
import pandas as pd
import csv

stemmer = EnglishStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
#    text = ''.join(ch for ch in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(ch) != 'Mn')
    text = "".join([ch for ch in text if ch not in (string.punctuation + '“'+ '”' + string.digits)])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def data_load():
    data_path=r'C:\\Users\Vishal Ghorpade\Desktop\20news-bydate.tar\20news-bydate\20news-bydate-train'
    list_classes=os.listdir(data_path)
    find_and_moveincompatible(data_path)
    files=sklearn.datasets.load_files(data_path,shuffle=True)
    print(list(files.target_names))
    data_clean(files.data)
    print(len(files.target))
    count_vect = CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize,lowercase=False)
    X_train_counts = count_vect.fit_transform(files.data)
    print(X_train_counts.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_train_tfidf, files.target, test_size=0.2)

    clf = MultinomialNB().fit(X_train, y_train)
    print(clf.score(X_train,y_train))
    print(clf.score(X_test,y_test))

    test_data_path=r'C:\Users\Vishal Ghorpade\Desktop\20news-bydate.tar\20news-bydate\20news-bydate-test'
    find_and_moveincompatible(test_data_path)
    test_data=sklearn.datasets.load_files(test_data_path)
    data_clean(test_data.data)
    data_counts= count_vect.transform(test_data.data)
    X_eval= tfidf_transformer.transform(data_counts)

    l1=['doc_id']
    l2=test_data.target_names
    header=l1+l2

    doc_id=[]
    df=pd.DataFrame(columns=header)
    for i in range(len(test_data.filenames)):
        d_id=(test_data.filenames[i]).split('\\')
        doc_id.append(d_id[len(d_id)-1])
    df['doc_id']=doc_id
    print(df.head())
    print(X_eval.shape)
    predicted = clf.predict_proba(X_eval)
    print(predicted[0])

    df[test_data.target_names]=np.asanyarray(predicted)
    print(df.head())
    df.to_csv('output_naive_bayes.csv')

def find_and_moveincompatible(data_path):
    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    check_files=sklearn.datasets.load_files(data_path)
    num = []
    for i in range(len(check_files.filenames)):
        try:
            count_vector.fit_transform(check_files.data[i:i + 1])
        except UnicodeDecodeError:
            num.append(check_files.filenames[i])
        except ValueError:
            pass

    for i in num:
        os.remove(i)


def data_clean(file_data):
    print(len(file_data))
    for i, text in zip(range(len(file_data)), file_data):
        #print(text)
        file_data[i] = clean_stem(text)



def clean_stem(text):
    data = text.decode().split('\n')
    text_data = []
    finished = False
    for part in data:
        if finished:
            text_data.append(part)
            continue
        if not (part.startswith('Path:') or part.startswith('Newsgroups:') or part.startswith('Xref:')) and not finished:
            text_data.append(part)
        if part.startswith('Lines:'):
            finished = True

    return text_data


if __name__ == "__main__":
    data_load()
