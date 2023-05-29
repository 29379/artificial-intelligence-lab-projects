from GlassType import GlassType

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.svm import SVC


"""
    PROBLEM - IDENTYFIKACJA RODZAJÓW SZKŁA
        1. Eksploracja danych                                                           |   10 pkt
            > Przedstawienie danych statystycznych i uwag 
                dotyczących cech i etykiet zbioru danych
        2. Przygotowanie danych                                                         |   30 pkt
            > Podział na zestaw uczący i testowy (walidacja krzyżowa - alternatywnie),
                badanie wpływu różnego rodzaju przetorzenia danych na wyniki
                klasyfikacji (np normalizacja vs standaryzacja vs dyskretyzacja
                vs selekcja cech vs PCA) - porównanie wyników bez przetworzenia 
                danych z wynikami po ich przetworzeniu z co najmniej 2 metodami
                różnego typu (osobno)
        3. Klasyfikacja                                                                |   40 pkt
            > Testowanie klasyfikatorów i zbadanie ich wpływu na wyniki:
                Bayes i drzewo decyzyjne z użyciem przynajmniej 3 różnych
                zestawów hiperparametrów  
        4. Ocena klasyfikacji                                                          |    20 pkt  
            > Porównanie wyników różnego typu przygotowania danych oraz 
                wykorzystanego klasyfikatora, ocena klasyfikacja, interpretacja
                wyników
        5. Raport
            > Opis wykonywanych kroków, opis rezultatów zadania (zebrane
                tabele), interpretacja, materiały źródłowe, opis bibliotek
                
"""

#   ------------------------------------------------------------------------------------------------

def read_data() -> pd.DataFrame:
    df = pd.read_csv("glass.data",      #   RI - refractive index - współczynnik załamania
                     header=None,       #   procenty wagowe związków w odpowiednim tlenku
                     delimiter=',',     #   typ szkła, wartości 1-7
                     names=['ID', 'RI', 'Na', 
                            'Mg', 'Al','Si',
                            'K','Ca','Ba',
                            'Fe','Type']
                     )
    return df


def get_X_and_Y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(['ID','Type'], axis=1)
    Y = df['Type']
    return X, Y


def data_exploration(df: pd.DataFrame) -> None:
    #   A chart to check out the count of different types of glass in the dataset
    sns.set(style="darkgrid", font_scale=1.5)    
    plt.subplots(figsize=(16,9))
    sns.countplot(data=df, x='Type').set_title('Count of different types of glass')
    plt.show()
    #   Type 1 i Type 2 - much more values

    #   A chart to check out the values of 'RI' and all the elements for all the glass types
    sns.set(style="darkgrid", font_scale=1.0)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 9))
    plt.subplots_adjust(hspace=0.33)
    sns.boxplot(x='Type', y='RI', data=df, ax=axes[0, 0])
    sns.boxplot(x='Type', y='Na', data=df, ax=axes[0, 1])
    sns.boxplot(x='Type', y='Mg', data=df, ax=axes[0, 2])
    sns.boxplot(x='Type', y='Al', data=df, ax=axes[1, 0])
    sns.boxplot(x='Type', y='Si', data=df, ax=axes[1, 1])
    sns.boxplot(x='Type', y='K', data=df, ax=axes[1, 2])
    sns.boxplot(x='Type', y='Ca', data=df, ax=axes[2, 0])
    sns.boxplot(x='Type', y='Ba', data=df, ax=axes[2, 1])
    sns.boxplot(x='Type', y='Fe', data=df, ax=axes[2, 2])
    """
        RI - similar values, but a much higher range in Type 5
        Na - higher values for types 6 & 7
        Mg - high values for types 1, 2, 3, high value ranges for types 5, 6, 7
        Al - high values for types 5 & 7
        Si - similar data
        K  - similar data
        Ca - high values in type 5, higher range in type 5
        Ba - very high values and very high value range in type 7
        Fe - much higher values and value ranges for types 1, 2 & 3
    """

    plt.show()


#   PCA - Principal Component Analysis, reducing the number of variables in a dataset
def split_dataset_with_PCA(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #   grab specific columns from the dataframe
    X_var = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
    pca = PCA(random_state=np.random.randint(100))   #   random seed
    pca.fit(X_var)
    
    var_exp = pca.explained_variance_ratio_ #   proportion of the variance explained by each component, e.g. RI, Na, Mg, ..., individual variance
    cum_var_exp = np.cumsum(var_exp)        #   cumulative explained variance
    
    var_df = pd.DataFrame(pca.explained_variance_.round(3), index=X_var.columns,
                        columns=["Explained_Variance"]) #   store the variance data in a pandas dataframe
    print(var_df.T) #   print the variance data to console

    #   create a figure, a bar and a plot
    plt.figure(figsize=(16, 9))
    plt.bar(X_var.columns, var_exp, align='center', label='individual variance', color='green')
    plt.plot(X_var.columns, cum_var_exp, marker='o', label='cumulative variance', color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='center right')
    
    # Add percentages to individual variance bars
    for i, v in enumerate(var_exp):
        plt.text(i, v + 0.02, f'{v*100:.1f}%', ha='center')

    # Add percentages to the cumulative variance line
    for i, v in enumerate(cum_var_exp):
        plt.text(i, v - 0.05, f'{v*100:.1f}%', ha='center', color='red')
    plt.show()
    
    #   RI, Na, Mg, Al, Si i K equals to more or less 99.8%, so it is safe to say that I can cut
    #   Ca, Ba i Fe out of the batch as it almost does not affect the data at all in this case
    pca_red = PCA(n_components=6)
    X_reduced = pca_red.fit_transform(X_var)
    
    #   Splitting the dataset with 80-20 proportions
    X = X_reduced
    Y = df['Type'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=np.random.randint(100))
    
    return x_train, x_test, y_train, y_test
    

def testing_models_SVC(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    svc_model = SVC()
    svc_model.fit(x_train, y_train)
    y_pred = svc_model.predict(x_test)
    svc_score = accuracy_score(y_pred, y_test)
    #print(svc_model.feature_importances_)
    return svc_score


def testing_models_DecisionTree(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    dec_tree_model = DecisionTreeClassifier()
    dec_tree_model.fit(x_train, y_train)
    y_pred = dec_tree_model.predict(x_test)
    dec_tree_score = accuracy_score(y_pred, y_test)
    #print(dec_tree_model.feature_importances_)
    return dec_tree_score
    

def testing_models_RandomForest(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    rand_forest_model = RandomForestClassifier(max_depth=3, min_samples_split=2, n_estimators=50, random_state=np.random.randint(100))
    rand_forest_model.fit(x_train, y_train)
    y_pred = rand_forest_model.predict(x_test)
    rand_forest_score = accuracy_score(y_pred, y_test)
    #print(rand_forest_model.feature_importances_)
    return rand_forest_score


def compare_scores(svc: float, dec_tree: float, rand_forest: float) -> None:
    scores = pd.DataFrame([['Support Vector Machine', svc],
                           ['Decision Tree', dec_tree],
                           ['Random Forest', rand_forest]])
    print(scores)


def hyperparameter_tuning_svc(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    tuned_parameters = [{'kernel': ['rbf','linear'], 'gamma': [0.001, 0.01, 0.1, 10],
                     'C': [0.001, 0.01, 0.1, 1, 10]}]
    grid = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy')
    grid.fit(x_train, y_train)
    
    print("Best set found on training datasets:")
    print(f' > {grid.best_params_}')
    print(f' > {grid.best_estimator_}')
    
    model = grid.best_estimator_
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    
    print('Score for best model and best parameter set: ')
    print(f' > {score}')
    
    matrix = confusion_matrix(y_test, y_pred)
    plt.subplots(figsize=(16, 9))
    sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    
    print(classification_report(y_test, y_pred))


def main():
    df = read_data()
    X, Y = get_X_and_Y(df)
    
    data_exploration(df)    
    x_train, x_test, y_train, y_test = split_dataset_with_PCA(df)
    svc = testing_models_SVC(x_train, x_test, y_train, y_test)
    dec_tree = testing_models_DecisionTree(x_train, x_test, y_train, y_test)
    rand_forest = testing_models_RandomForest(x_train, x_test, y_train, y_test)
    
    compare_scores(svc, dec_tree, rand_forest)
    print('-----')
    hyperparameter_tuning_svc(x_train, x_test, y_train, y_test)








if __name__ == '__main__':
    main()
    