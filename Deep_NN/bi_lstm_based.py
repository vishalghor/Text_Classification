import re
import keras
import numpy as  np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam,RMSprop
from keras.layers.normalization import BatchNormalization
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from keras import metrics
from keras import regularizers,initializers



line_codes=[]
code_list=[]
codes=[]
cat=[]
tokenizer = WordPunctTokenizer()
vocab = Counter()
line_codes_test=[]
code_list_test=[]
codes_test=[]




def data_load(path):
    with open(r'training-data-small.txt','r') as fp:
        readlines=fp.readlines()
        readlines=[line.rstrip('\n') for line in readlines]
        for i in range(len(readlines)):
            line_codes=(readlines[i].split('\t')[1])
            line_codes = re.sub(',', ' ', line_codes)

            text = tokenizer.tokenize(line_codes)
            cat.append(readlines[i].split('\t')[0])
            code_list.append(text)
            vocab.update(text)
            for j in range(len(text)):
                codes.append(text[j])

    return readlines



def data_load_test(path):
    with open(path,'r') as fp:
        readlines_test=fp.readlines()
        readlines_test=[line.rstrip('\n') for line in readlines_test]
        for i in range(len(readlines_test)):
            line_codes_test=(readlines_test[i])
            line_codes_test = re.sub(',', ' ', line_codes_test)

            text = tokenizer.tokenize(line_codes_test)
            code_list_test.append(text)
            vocab.update(text)
            for j in range(len(text)):
                codes_test.append(text[j])

    return readlines_test



def word_vec(code_list):
    model = Word2Vec(code_list, size=100, window=5, min_count=5, workers=16, sg=0, negative=5)
    word_vectors = model.wv
    print("Number of word vectors: {}".format(len(word_vectors.vocab)))
    return word_vectors




def main():
    readlines=data_load(r'training-data-large.txt')
    readlines_test=data_load_test(r'test-data-small.txt')

    print(code_list)
    print(len(code_list))
    print(np.unique(codes))
    print(len(np.unique(codes)))
    categories =np.unique(cat)
    print(len(cat))
    print(categories)
    y=np.asarray(cat)
    print(y)
    data_list=code_list+code_list_test
    word_vectors=word_vec(data_list)
    MAX_NB_WORDS = len(word_vectors.vocab)
    MAX_SEQUENCE_LENGTH =len(np.unique(codes+codes_test))
    print(len(code_list[1]))
    word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(MAX_NB_WORDS))}
    sequences = [[word_index.get(t, 0) for t in comment] for comment in code_list[:len(readlines)]]

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    y=keras.utils.to_categorical(y)
    print(data[0])
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)

    train_data=data
    #test_data=data[]
    y_train=y
    #y_test=y[n:]
    print(y_train)
    WV_DIM = 100
    nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab)) +1
    # we initialize the matrix with random numbers
    wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
    for i in range(len(word_index)):
        if i >= MAX_NB_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word_index[i]]
            # words not found in embedding index will be all-zeros.
            wv_matrix[i] = embedding_vector
        except:
            pass



    embedding_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=True)

    # Inputs
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    print(comment_input.shape)
    embedded_sequences = embedding_layer(comment_input)

    embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(LSTM(64,kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),kernel_regularizer=regularizers.l1_l2(0.01,0.01),return_sequences=False))(embedded_sequences)

    # Output
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    preds = Dense(2, activation='softmax')(x)

    # build the model
    model = Model(inputs=[comment_input], outputs=preds)
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.2),#Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
              metrics=[metrics.categorical_accuracy])

    #X_train, X_test, y_train, y_test=sklearn.model_selection.train_test_split(code_list, cat, test_size=0.2)

    model.fit([train_data],y_train,validation_split=0.1,epochs=10, batch_size=256, shuffle=True)

    #test data

    #word_index_test = {t[0]: i+1 for i,t in enumerate(vocab.most_common(MAX_NB_WORDS))}
    sequences_test = [[word_index.get(t, 0) for t in comment] for comment in code_list_test[:len(readlines_test)]]

    data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    print(data_test[0])
    print('Shape of data tensor:', data_test.shape)
    y_test_pred=model.predict(data_test)
    y_test_pred=np.where(y_test_pred>0.5,1,0)
    print(y_test_pred)
    res_test=[]
    for line in (y_test_pred):
        r=categories[np.argmax(line)]
        res_test.append(r)
    print(res_test)

    file_pred=open('small_test_pred.txt','w')
    for item in res_test:
        file_pred.write("%s\n" % item)


if __name__=="__main__":
    main()
