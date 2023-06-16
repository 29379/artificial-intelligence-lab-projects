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
from itertools import product


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
    # x_train, x_test, y_train, y_test = train_test_split(
    #     embeeded_jokes,
    #     ratings,
    #     test_size=.2,
    #     random_state=1
    # )
    # x_train_scaled = scaler.fit_transform(x_train)
    # x_test_scaled = scaler.fit_transform(x_test)
    
    normalized_embeedings = scaler.fit_transform(embeeded_jokes)
    x_train, x_test, y_train, y_test = train_test_split(
        normalized_embeedings, ratings, test_size=.2, random_state=3
    )
    
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    
    return x_train, x_test, y_train, y_test


# def run_MLPRegressor(x_train: any, x_test: any, y_train: any, y_test: any) -> MLPRegressor:
#     model = MLPRegressor(
#         hidden_layer_sizes=(100, 100, 100),
#         activation='relu',
#         solver='adam',
#         max_iter=500,
#         random_state=1
#     )
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     return model
def run_regressor(x_train: any, x_test: any, y_train: any, y_test: any, params: dict) -> tuple[list[float], list[float], list[float]]:
    _hidden_layer_sizes = params['hidden_layer_sizes']
    _activation = params['activation']
    _alpha = params['alpha']
    _random_state = params['random_state']
    _max_iter = params['max_iter']
    _solver = params['solver']
    _learning_rate = params['learning_rate']
    _learning_rate_init = params['learning_rate_init']
    _epochs = params['epochs']
    
    mlp = MLPRegressor(
        hidden_layer_sizes=_hidden_layer_sizes,
        activation=_activation,
        solver=_solver,
        max_iter=_max_iter,
        random_state=_random_state,
        learning_rate=_learning_rate,
        learning_rate_init=_learning_rate_init,
        alpha=_alpha
    )
    
    train_loss = []
    test_loss = []
    
    for epoch in range(_epochs):
        mlp.partial_fit(x_train, y_train)
        pred_y_train = mlp.predict(x_train)
        pred_y_test = mlp.predict(x_test)
        
        train_loss.append(mean_squared_error(y_train, pred_y_train))
        test_loss.append(mean_squared_error(y_test, pred_y_test))
    loss_curve = mlp.loss_curve_    
    return train_loss, test_loss, loss_curve
    
            
def run_with_constant_learning_rates(x_train: any, x_test: any, y_train: any, y_test: any) -> None:
    params: list[dict] = [
        {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'alpha': 0.0,
            'random_state': 0,
            'max_iter': 500,
            'solver': 'sgd',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'epochs': 500
        },
        {
            'hidden_layer_sizes': (100,100,),
            'activation': 'relu',
            'alpha': 0.0,
            'random_state': 1,
            'max_iter': 500,
            'solver': 'sgd',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'epochs': 500
        },
        {
            'hidden_layer_sizes': (100,100,100),
            'activation': 'relu',
            'alpha': 0.0,
            'random_state': 1,
            'max_iter': 500,
            'solver': 'sgd',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'epochs': 500
        },
        {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'alpha': 0.0,
            'random_state': 1,
            'max_iter': 500,
            'solver': 'sgd',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'epochs': 1000
        },
        {
            'hidden_layer_sizes': (100,100,),
            'activation': 'relu',
            'alpha': 0.0,
            'random_state': 1,
            'max_iter': 500,
            'solver': 'sgd',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'epochs': 1000
        },
        {
            'hidden_layer_sizes': (100,100,100),
            'activation': 'relu',
            'alpha': 0.0,
            'random_state': 1,
            'max_iter': 500,
            'solver': 'sgd',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'epochs': 1000
        }
    ]
    
    for param_batch in params:
        loss_train, loss_test, loss_curve = run_regressor(x_train, x_test, y_train, y_test, param_batch, mode['train'])
        plt.title(f'Epochs: {param_batch["epochs"]} | Layers: {param_batch["hidden_layer_sizes"]}')
        plt.plot(range(len(loss_train)), loss_train, label=f'Train Loss')
        plt.plot(range(len(loss_test)), loss_test, label=f'Test Loss')
        plt.plot(range(len(loss_curve)), loss_curve, label=f'Loss curve')
        plt.legend()
        plt.show()
        

def run(x_train: any, x_test: any, y_train: any, y_test: any) -> None:
    solver = ['sgd']
    activation = ['relu']
    alpha = [0.0]
    random_state = [0]
    max_iter = [500]
    
    hidden_layer_sizes=[(25,), (25, 25,),
                        (100,),(100, 100,),
                        (500,), (500, 500)]
    learning_rates=['constant', 'adaptive', 'invscaling']
    learning_rate_inits=[0.001, 0.01, 0.1, 1.]
    epochs=[100, 500, 1000, 2000]
    
    parameter_combinations = list(product(hidden_layer_sizes, learning_rates, learning_rate_inits, epochs))
    
    for combination in parameter_combinations:
        hidden_layer_size, learning_rate, learning_rate_init, epoch = combination
        params = {
            'hidden_layer_sizes': combination[0],
            'learning_rate': combination[1],
            'learning_rate_init': combination[2],
            'epochs': combination[3],
            'activation': 'relu',
            'alpha': 0.0,
            'random_state': 0,
            'max_iter': 500,
            'solver': 'sgd',
        }
        print('-------------------\n')
        print(f'Running with params: {params}')
        print('\n-------------------\n')
        loss_train, loss_test, loss_curve = run_regressor(x_train, x_test, y_train, y_test, params)
        plt.title(f'Epochs: {params["epochs"]} | Layers: {params["hidden_layer_sizes"]} | Learning rate: {params["learning_rate"]} | Learning rate init: {params["learning_rate_init"]}')
        plt.plot(range(len(loss_train)), loss_train, label=f'Train Loss')
        plt.plot(range(len(loss_test)), loss_test, label=f'Test Loss')
        plt.plot(range(len(loss_curve)), loss_curve, label=f'Loss curve')
        plt.legend()
        plt.show()
    

if __name__ == '__main__':    
    ratings = read_ratings()
    jokes = read_jokes()
    embeeded_jokes = generate_embeedings(jokes)
    x_train, x_test, y_train, y_test = split_dataset_with_StandardScaler(embeeded_jokes, ratings)
    #run_with_constant_learning_rates(x_train, x_test, y_train, y_test)
    run(x_train, x_test, y_train, y_test)

    
