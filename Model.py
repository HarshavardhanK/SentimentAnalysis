#Sentiment Analysis using python with IMDb movie review dataset
#also called opinioin mining

import pyprind
import pandas as pd
import os
import shelve

#----------------------------------- DATASET HANDLING --------------------------

dataset_path = "/Users/HarshavardhanK/Desktop/Code Files/Sublime/Python/MachineLearning/SentimentAnalysis/Datasets"

pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}

def populate_dataset():

    df = pd.DataFrame()

    for s in ('test', 'train'):
        for l in ('pos', 'neg'):

            path = "/Users/HarshavardhanK/Desktop/Code Files/Datasets/SentimentAnalysis/aclImdb/%s/%s" % (s, l)

            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()

                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()

    df.columns = ['review', 'sentiment']

    dataset = shelve.open(dataset_path)
    dataset['movieReview'] = df
    dataset.close()

    print('Done populating dataset')

def load_dataset():

    df = pd.DataFrame()

    dataset = shelve.open(dataset_path)
    df = dataset['movieReview']
    dataset.close()

    print('Done loading dataset')

    return df

#populate_dataset()
df = load_dataset()

#-------------------------------------------------------------------------------

import numpy as np

np.random.seed(0)

df = df.reindex(np.random.permutation(df.index))
new_path = os.path.join("/Users/HarshavardhanK/Desktop/Code Files/Sublime/Python/MachineLearning/SentimentAnalysis", '/movie_data.csv')
df.to_csv('./movie_data.csv', index=False)
df = pd.read_csv('./movie_data.csv')
df.head(5)
