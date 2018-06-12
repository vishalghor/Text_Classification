import pandas as pd
import nltk
nltk.download('stopwords')
import re
import string

pd.set_option('display.max_colwidth', 150)
fullCorpus = pd.read_csv(r'C:\\Sheet_2.csv',usecols=['resume_id','class','resume_text' ],encoding='latin-1')
fullCorpus.columns = ['','label', 'body_text']
print(fullCorpus.head())
print(fullCorpus['label'].value_counts())
print("Input data has {} rows and {} columns".format(len(fullCorpus), len(fullCorpus.columns)))
print("Out of {} rows, {} are flagge, {} are not_flagged".format(len(fullCorpus),
                                                       len(fullCorpus[fullCorpus['label']=='flagged']),
                                                       len(fullCorpus[fullCorpus['label']=='not_flagged'])))

print("Number of null in label: {}".format(fullCorpus['label'].isnull().sum()))
print("Number of null in text: {}".format(fullCorpus['body_text'].isnull().sum()))

def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

fullCorpus['body_text_clean'] = fullCorpus['body_text'].apply(lambda x: remove_punct(x))

print(fullCorpus.head())


def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

fullCorpus['body_text_tokenized'] = fullCorpus['body_text_clean'].apply(lambda x: tokenize(x.lower()))

print(fullCorpus.head())


stopword = nltk.corpus.stopwords.words('english')
ps=nltk.PorterStemmer()
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

fullCorpus['body_text_nostop'] = fullCorpus['body_text_tokenized'].apply(lambda x: remove_stopwords(x))

print(fullCorpus.head())


def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

fullCorpus['body_text_stemmed'] = fullCorpus['body_text_nostop'].apply(lambda x: stemming(x))

print(fullCorpus.head())