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


def read_ratings() -> pd.DataFrame:
    ratings = pd.DataFrame()
    for i in [1, 2, 3]:
        ratings_df = pd.read_excel(f'ratings/jester-data-{i}.xls', header=None)
        ratings_df = ratings_df.iloc[:, 1:].replace(99, float('nan'))
        ratings = pd.concat([ratings_df, ratings], ignore_index=True)

    ratings = ratings_df.mean() #   mean values to give one unified score for every joke
    return ratings


def read_jokes() -> list[str]:
    jokes = []
    for i in range(1, 101):
        file_name = f'jokes/init{i}.html'
        with open(file_name, 'r') as file:
            html_code = file.read()
            soup = BeautifulSoup(html_code, 'html.parser')
            joke = soup.find('font', size='+1').text.strip()
            jokes.append(joke)

    return jokes


def generate_embeedings(jokes: list[str]) -> any:
    model = SentenceTransformer('bert-base-cased')
    embeddings = model.encode(jokes)    #   encoding the jokes and changing text values into vectors
    print(embeddings.shape)
    return embeddings


def split_dataset_with_StandardScaler(embeeded_jokes: any, ratings: pd.DataFrame) -> tuple[any, any, any, any]:
    scaler = StandardScaler()
    x_train, x_test, y_train, y_test = train_test_split(
        embeeded_jokes,
        ratings,
        test_size=.2,
        random_state=1
    )
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    
    return x_train_scaled, x_test_scaled, y_train, y_test
    


if __name__ == '__main__':
    ratings = read_ratings()
    jokes = read_jokes()
    embeeded_jokes = generate_embeedings(jokes)
    x_train, x_test, y_train, y_test = split_dataset_with_StandardScaler(embeeded_jokes, ratings)


    
