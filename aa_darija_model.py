import os
import pandas as pd
import re
from fextractorRefactored import get_grams, getcount_ksngrams
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time

data_path_darija = './data/'
authors_darija = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

# Loading author tweets and target values into lists
def group_data(path, authors, nb_authors):
    chunks = ['01.txt', '02.txt', '03.txt', '04.txt', '05.txt', '06.txt', '07.txt', '08.txt', '09.txt', '10.txt']
    all_tweets = []
    y = []
    nb_authors = nb_authors - 1
    # chosing the range of authors
    for i in range(0, nb_authors):
        # path for author data    
        authors_path = path+str(authors[i])+"/"
        # list of author chunked data
        author = os.listdir(authors_path)
        # print(authors_path)
        for doc in range(len(chunks)):
            # path for chunked data 
            file = authors_path+str(chunks[doc])
            # print(file)
            # get data from file
            with open(file, 'r', encoding="utf8") as f:
                # list of tweets from chunked data
                tweets = [line.strip() for line in f]
                # add tweets to global tweets list
                all_tweets = all_tweets + tweets
                # target variable of data chunk
                y_chunk = [authors[i]]*len(tweets)
                # add target varianle list to global list
                y = y + y_chunk

        # returning list of all tweets with list of target values (authors)
        return all_tweets, y

# clean tweets data
def clean_data_list(data):
    # remove punctuation
    rm_punct_data = [re.sub(r'[^\w\s]', '', tweet) for tweet in data]
    # remove digits
    rm_digits_data = [" ".join([i for i in tweet.split(" ") if not i.isdigit()]) for tweet in rm_punct_data]

    # returning clean list of tweets
    return rm_digits_data

# getting tweets grams
def get_grams_data_list(data, gram):
    grammed_tweets = [get_grams(tweet, gram) for tweet in data]

    return grammed_tweets

# join grams tokens into str
def join_tokens(data):
    data_tweets = [' '.join(i[0]) for i in data]

    return data_tweets

# prepare data (applying all function above)
def prepare_tweets(data_path_darija, authors_darija, gram, nb_authors):
    # Get data
    data, target = group_data(data_path_darija, authors_darija, nb_authors)
    print("Data grouped ... ")
    # Clean data
    data = clean_data_list(data)
    print("Data Cleaned ...")
    # Get grams
    grammed_data = get_grams_data_list(data, gram)
    print("Data grammed ...")
    # Join grams tokens
    data_grm = join_tokens(grammed_data)
    print("Data joined ...")

    return data_grm, target

# getting k-skip n-grams vectorization of text
def vect_grams(data):
  # get k-skip-ngrams with their count from grams
  k_skip_data = [getcount_ksngrams(i, k=0, n=1, normalize=True) for i in data]
  # the keys are already the tokens in grams(data) so no need to extract them
  # we will extract the count of the kskip ngram into a list
  all_vect_data = []
  for g in range(0, data):
    vect_text = []
    for k in data[g][0]:
      if k in k_skip_data[g]:
        vect_text.append(k_skip_data[g][k])

    all_vect_data.append(vect_text)
  
  return all_vect_data

def aa_rfs_model(data, target):
    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2)
    # prepare data
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    train_data = vectorizer.transform(X_train).toarray()
    test_data = vectorizer.transform(X_test).toarray()
    print("data vectorized")
    # Build classifier
    clf = RandomForestClassifier(n_estimators=300)
    # train model
    clf.fit(train_data, Y_train)
    print("Model trained")
    # evaluate model
    # Get test predictions
    testPredictions = clf.predict(test_data)
    testPredictionsProbs = clf.predict_proba(test_data)

    # Calculate metrics
    accuracy = round(accuracy_score(Y_test, testPredictions) * 100, 2)
    precision = round(precision_score(Y_test, testPredictions, average = 'macro') * 100, 2)
    recall = round(recall_score(Y_test, testPredictions, average = 'macro') * 100, 2)
    fscore = round(f1_score(Y_test, testPredictions, average = 'macro') * 100, 2)
    confusionMatrix = confusion_matrix(Y_test, testPredictions)
    # display results
    # print("Accuarcy: ", accuracy)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F-score: ", fscore)

    print("Accuracy: {}% | Precision: {}% | Recall: {}% | F-score: {}".format(accuracy, precision, recall, fscore))
    print("Confusion matrix: \n", confusionMatrix)


# Implementation

# gram: pos, word, word-pos
gram = 'word-pos'
nb_authors = 4
tic = time.time()
# prepare tweets data
data, target = prepare_tweets(data_path_darija, authors_darija, gram, nb_authors)
# print(data[:10])
aa_rfs_model(data, target)
toc = time.time()
print("Execution time:", str(1000*(toc-tic)) + "ms")