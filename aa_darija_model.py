# imports 
import numpy as np
import pandas as pd
import itertools
from fextractorRefactored import get_grams, getcount_ksngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import os
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Loading author tweets and target values into lists
def group_data_tweet(data_path, authors, nb_authors):
    all_tweets = []
    y = []
    # authors
    for author in authors:
        # authors path
        text_files_path = data_path+author+"/"
        # all text files of selected authors
        chunks = listdir(text_files_path)
        for text_file in chunks:
            file = text_files_path+text_file
            with open(file, 'r', encoding="utf8") as f:
            # list of tweets from chunked data
            tweets = [line.strip() for line in f]
            # add tweets to global tweets list
            all_tweets = all_tweets + tweets
            # target variable of data chunk
            y_chunk = [authors[i]]*len(tweets)
            # add target varianle list to global list
            y = y + y_chunk

    return text_data, y

# Loading author chunks and target values into lists
def group_data_folder(data_path, authors, nb_authors):
    text_data = []
    y = [] 
    # authors
    # authors = listdir(data_path)[:nb_authors]
    for author in authors:
        # authors path
        text_files_path = data_path+author+"/"
        # print(text_files_path)
        # all text files of selected authors
        chunks = listdir(text_files_path)
        for text_file in chunks:
            file = text_files_path+text_file
            with open(file, 'r', encoding="utf8") as f:
            text_data.append(f.read())
            # print(text_data)
            # print(len(text_data))
            y.append(author)
  
    return text_data, y
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

# prepare data (applying all function above)
def prepare_tweets(data_path_darija, authors_darija, gram, nb_authors):
    # Get data
    data, target = group_data(data_path_darija, authors_darija, nb_authors)
    print("Data grouped ... ")
    # Clean data is not needed - that's why the lines below are commented
    # data = clean_data_list(data)
    # print("Data Cleaned ...")
    # Get grams
    grammed_data = get_grams_data_list(data, gram)
    print("Data grammed ...")
    # Join grams tokens
    # data_grm = join_tokens(grammed_data)
    # print("Data joined ...")

    return data_grm, target

# getting k-skip n-grams vectorization of text
def vect_norm_grams(data, skips, ng, l):
    # get k-skip-ngrams with their count from grams
    k_skip_data = [getcount_ksngrams(i, k=skips, n=ng, minfreq=l, normalize=True) for i in data]
    # print(k_skip_data)
    # get the keys of the dictionary
    features_key = {}
    for g in range(0, len(data)):
        for k in k_skip_data[g].keys():
            # Storing all keys (gram tokens)
            features_key[k] = 0


    # print(len(features_key.keys()))
    returned_data = []
    X = []

    for g in range(0, len(data)):
        X_row = []
        for f in features_key.keys():
            if f in k_skip_data[g]:
            # print(f)
            tkn_gram = k_skip_data[g][f]
            # print(tkn_gram)
            X_row.append(tkn_gram)
            else:
            X_row.append(0)
            
        X.append(X_row)
    # print("Features length: ", len(X))

    return X

# return the accuracies for all authors combinations of input parameters
def evaluate_combination(data_path, k, l, n, nb_authors, gram):

    # list params
    k_list = []
    l_list = []
    n_list = []
    grams_list = []
    nb_authors_list = []

    # metrics
    accuracy_list = []
    precision_list = []
    recall_list = []
    fscore_list = []

    grams_type = [gram]

    # authors
    authors_uae = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

    # authors combinations
    authors_combinations = list(itertools.combinations(authors_uae, nb_authors))
    print("Authors space: ", nb_authors,"| Total combinations: ", len(authors_combinations))

    # looping through the authors combination to train the model
    for combination in range(0, len(authors_combinations)):
        authors_train = list(authors_combinations[combination])
        print("Authors to train: ",authors_train, " | Combination number: ", combination+1, " | Combinations left: ", len(authors_combinations)-(combination+1))

        # group data
        data, target = group_data(data_path, authors_train, nb_authors)
        print("Data grouped | Total authors: ", len(authors_train), " | Authors: ", authors_train)

        for gram in grams_type:
            print("Append K, L, N, gram and number of authors")
            grams_list.append(gram)
            l_list.append(l)
            n_list.append(n)
            k_list.append(k)
            nb_authors_list.append(nb_authors)

            print("Gramming data: ", gram)
            grammed_data = get_grams_data_list(data, gram)
            print("Grammed data: ", len(grammed_data), " | Target: ", len(target))
            print("Grammed tweet 1:", grammed_data[0])

            # vectorize data
            print("Vectorizing data")
            features_data = vect_norm_grams(grammed_data, k, n, l)
            print("Data vectorized")

            # building model
            print("__________________________________________________________")
            print("Gram={}, k={}, n={}, l={}, nb_autors={}".format(gram, k, n, l, nb_authors))
            # split data
            print("Data splitting ...")
            X_train, X_test, Y_train, Y_test = train_test_split(features_data, target, test_size=0.2)

            # model tarining
            print("Training model ...")
            clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
            clf.fit(X_train, Y_train)

            # evaluate 
            print("Evaluating ...")
            testPredictions = clf.predict(X_test)
            # testPredictionsProbs = clf.predict_proba(X_test)
            # Calculate metrics
            # Calculate metrics
            accuracy = round(accuracy_score(Y_test, testPredictions) * 100, 2)
            precision = round(precision_score(Y_test, testPredictions, average = 'macro') * 100, 2)
            recall = round(recall_score(Y_test, testPredictions, average = 'macro') * 100, 2)
            fscore = round(f1_score(Y_test, testPredictions, average = 'macro') * 100, 2)
            confusionMatrix = confusion_matrix(Y_test, testPredictions)

            # save metircs
            fscore_list.append(fscore)
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
            print("Metrics added ...")
            print(accuracy_list)

    print("Create dataframe of results ... ")
    df = pd.DataFrame(list(zip(l_list, k_list, n_list, grams_list, nb_authors_list, accuracy_list, precision_list, recall_list, fscore_list)),
                columns =['l', 'k', 'n', 'Gram', 'Total Authors', 'Accuracy', 'Precision', 'Recall', 'F-score'])

    print("Save dataframe of results ... ")
    df.to_csv("./evaluation/authors"+str(nb_authors)+"_k"+str(k)+"_n"+str(n)+"_l"+str(l)+"_"+gram+".csv")
    print("Datframe saved ...")
    print(df)

# Implementation
data_path = './data/'
n = 1
k = 0
l = 1
nb_authors = 4
gram = 'word'
evaluate_combination(data_path, k, l, n, nb_authors, gram)