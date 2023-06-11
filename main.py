"""
The text of the jokes are in the 'jokes' folder
Format: 'init'x'.html', x in range [1, 100], where x corresponds to the ID of the column in files with ratings

The Ratings Data: jester-data-1/2/3.xls
Ratings: values in range [-10, 10], 99 means null value (no rating). 
Rows - users
Columns - ratings for each joke
First column - number of jokes rated by that user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from enum import Enum


def generate_embeedings() -> None:
    model = SentenceTransformer('bert-base-cased')
    texts_list = ["Hello, my dog is cute.", "I love Artificial Intelligence. Machine Learning is my passion!"]
    embeddings = model.encode(texts_list)
    print(embeddings.shape)

def read_ratings() -> pd.DataFrame:
    ratings = pd.DataFrame()
    for i in [1, 2, 3]:
        ratings_df = pd.read_excel(f'ratings/jester-data-{i}.xls', header=None)
        ratings_df = ratings_df.iloc[:, 1:].replace(99, float('nan'))
        ratings = pd.concat([ratings_df, ratings], ignore_index=True)

    ratings = ratings_df.mean() #   mean values to give one unified score for every joke
    return ratings


joke_texts = 'jokes/Dataset4JokeSet.xlsx'
ratings = 'ratings/[final] April 2015 to Nov 30 2019 - Transformed Jester Data.xlsx'

#   jokes:      1 column (the joke)
#   ratings:    rows - user ratings, columns - joke ratings
def read_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    joke_texts_df = pd.read_excel(joke_texts)
    ratings_df = pd.read_excel(ratings)
    
    #   99 means no rating, so I dropped all the rows and columns, where it is appropriate, where there are only 99s
    columns_without_any_rating = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 20, 27, 31, 43, 51, 52, 61, 73, 80, 100, 116]
    rows_to_drop = [joke - 1 for joke in columns_without_any_rating]
    joke_texts_df.drop(rows_to_drop)
    ratings_df.drop(ratings_df.columns[columns_without_any_rating], axis=1, inplace=True)
    
    return joke_texts_df, ratings_df

if __name__ == '__main__':
    # jokes, ratings = read_files()
    # indexes = jokes.loc[(jokes == 99).all(axis=1)].index
    # print(indexes)
    
    # indexes = ratings.columns[(ratings == 99).all(axis=0)]
    # print(indexes)
    
    # generate_embeedings()
    read_ratings()


    
