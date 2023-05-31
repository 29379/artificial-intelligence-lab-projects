from GlassType import GlassType

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif


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

#   ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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


#   ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


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
    
    X_var = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
    pca = PCA(random_state=1)   #   seed
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


#   ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#   PCA - Principal Component Analysis, reducing the number of variables in a dataset
#   and keeping the most important info. 
def split_dataset_with_PCA(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #   grab specific columns from the dataframe
    X_var = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
    pca = PCA(random_state=1)   #   seed
    pca.fit(X_var)
    
    #   RI, Na, Mg, Al, Si i K equals to more or less 99.8%, so it is safe to say that I can cut
    #   Ca, Ba i Fe out of the batch as it almost does not affect the data at all in this case
    pca_red = PCA(n_components=6)
    X_reduced = pca_red.fit_transform(X_var)
    
    X = X_reduced
    Y = df['Type'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1)
    return x_train, x_test, y_train, y_test
    
    
#   Similar to PCA, but: instead of a linear transformation into a new batch of uncorrelated features it performs
#   ANOVA variance analysis and scores the importance of each feature, giving a batch of the most important
#   features without reducing the dimensions. PCA also works on the basis of maximizing the variance within the
#   dataset, while this feature selection chooses on the basis of what is most influential for the classification quality
def split_dataset_with_FeatureSelection(df: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #   grab specific columns from the dataframe
    X = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
    Y = df['Type'].values
    feature_selector = SelectKBest(f_classif, k=k)  #   f_classif - feature importance measure
    x2 = feature_selector.fit_transform(X, Y)
    
    x_train, x_test, y_train, y_test = train_test_split(x2, Y, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test    


#   Standarization - deleting the mean and scaling to unit variance, calculates mean and
#   standard deviation  and scales it in a way, that makes mean 0 and deviation 1. 
#   Standard scaler assumes that the distribution of features is close to Gaussian distribution
#   and is appropriate, when data does not have a specific range and can have negative values
def split_dataset_with_StandardScaler(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #   grab specific columns from the dataframe
    X = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
    Y = df['Type'].values
    scaler = StandardScaler()
    x2 = scaler.fit_transform(X)
    
    x_train, x_test, y_train, y_test = train_test_split(x2, Y, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test


#   MinMaxScaler - min-max scaling is used to scale features in a specific range, usually 0 --> 1.
#   It calculates min and max for every feature and the scales it accordingly to fit into the range.
#   MinMax keeps the shape of the deature distribution and is appropriate even if it is not Gaussian, 
#   and when features have a specific value range
def split_dataset_with_MinMaxScaler(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #   grab specific columns from the dataframe
    X = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
    scaler = MinMaxScaler()
    x2 = scaler.fit_transform(X)
    
    x_train, x_test, y_train, y_test = train_test_split(x2, df['Type'].values, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test


#   Normalization - scaling every row separately in a way, that makes them have unit norm, which means
#   that the squared sum of the values is 1. Used when the direction/angle of the data matters more than
#   specific feature magnitudes. Brings features to a similar scale, removes the dominance of a single feature
def split_dataset_with_Normalizer(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #   grab specific columns from the dataframe
    X = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
    normalizer = Normalizer()
    x2 = normalizer.fit_transform(X)
    
    x_train, x_test, y_train, y_test = train_test_split(x2, df['Type'].values, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test    


#   Transforming constant variables into disccrete ones, splitting data into ranges or categories.
#   Uniform discretization splits the values into equal ranges. Encoding is ordinal instead of default one-hot,
#   because it preserves the order of the categories
def split_dataset_with_Discretizer(df: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Grab specific columns from the dataframe
    X = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    x2 = discretizer.fit_transform(X)
    
    x_train, x_test, y_train, y_test = train_test_split(x2, df['Type'].values, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test


def split_dataset_without_preprocessing(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Grab specific columns from the dataframe
    X = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
    Y = df['Type'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test


#   ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def testing_models_SVC(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float, float]:
    svc_model = SVC()
    svc_model.fit(x_train, y_train)
    y_pred = svc_model.predict(x_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    
    return acc, prec, rec, f1


def testing_models_DecisionTree(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float, float]:
    dec_tree_model = DecisionTreeClassifier()
    dec_tree_model.fit(x_train, y_train)
    y_pred = dec_tree_model.predict(x_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    
    return acc, prec, rec, f1
    

def testing_models_RandomForest(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float, float]:
    rand_forest_model = RandomForestClassifier(max_depth=3, min_samples_split=2, n_estimators=50, random_state=1)
    rand_forest_model.fit(x_train, y_train)
    y_pred = rand_forest_model.predict(x_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    
    return acc, prec, rec, f1


def testing_models_GaussianNB(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float, float]:
    gaussian_nb_model = GaussianNB()
    gaussian_nb_model.fit(x_train, y_train)
    y_pred = gaussian_nb_model.predict(x_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    
    return acc, prec, rec, f1


def compare_scores(svc_scores: float, dec_tree_scores: float, rand_forest_scores: float, gaussian_nb_scores: float) -> None:
    scores = pd.DataFrame([
        ['Support Vector Machine'] + list(svc_scores),
        ['Decision Tree'] + list(dec_tree_scores),
        ['Random Forest'] + list(rand_forest_scores),
        ['Gaussian Naive Bayes'] + list(gaussian_nb_scores)
    ], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    print('\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n')
    print("          SVC, DEC_TREE, RAND_FOREST AND BAYES SCORES             ")
    print(scores)
    print('\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n')


#   ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def hyperparameter_tuning_svc(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    tuned_parameters = [
        {'kernel': 'rbf', 'gamma': 0.001, 'C': 1},
        {'kernel': 'linear', 'C': 0.1},
        {'kernel': 'rbf', 'gamma': 0.01, 'C': 0.1},
        {'kernel': 'linear', 'C': 1},
        {'kernel': 'rbf', 'gamma': 0.1, 'C': 0.01}
    ]
    scores = []

    for params in tuned_parameters:
        model = SVC(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        scores.append([acc, prec, rec, f1, params])


    cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Parameters']
    scores_df = pd.DataFrame(scores, columns=cols)
    
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")
    print("               SVC WITH HYPERTUNED PARAMETERS   ")
    print(scores_df)
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")
    return scores_df
    

def hyperparameter_tuning_decision_tree(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    tuned_parameters = [
        {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 4},
        {'criterion': 'entropy', 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 2}
        ]
    scores = []
    
    for params in tuned_parameters:
        model = DecisionTreeClassifier(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        scores.append([acc, prec, rec, f1, params])
    
    cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Parameters']
    scores_df = pd.DataFrame(scores, columns=cols)
    
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")
    print("            DecisionTree WITH HYPERTUNED PARAMETERS   ")
    print(scores_df)
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")
    return scores_df


def hyperparameter_tuning_naive_bayes(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    tuned_parameters = [
        {'var_smoothing': 1e-9},
        {'var_smoothing': 1e-8},
        {'var_smoothing': 1e-7},
        {'var_smoothing': 1e-6},
        {'var_smoothing': 1e-5}
    ]
    scores = []
    
    for params in tuned_parameters:
        model = GaussianNB(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        scores.append([acc, prec, rec, f1, params])
    
    cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Parameters']
    scores_df = pd.DataFrame(scores, columns=cols)
    
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")
    print("            Naive Bayes WITH HYPERTUNED PARAMETERS   ")
    print(scores_df)
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")
    return scores_df


#   ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def test_models(df: pd.DataFrame) -> None:
    # Initialize an empty DataFrame
    results_df = pd.DataFrame()

    for preprocess_name, preprocess_method in [("PCA", split_dataset_with_PCA),
                                               ("Feature Selection", split_dataset_with_FeatureSelection),
                                               ("Standarization", split_dataset_with_StandardScaler),
                                               ("Normalization", split_dataset_with_Normalizer),
                                               ("Discretization", split_dataset_with_Discretizer),
                                               ("MinMax Scaling", split_dataset_with_MinMaxScaler),
                                               ("No Preprocessing", split_dataset_without_preprocessing)]:
        

        if preprocess_name == "Feature Selection":
            x_train, x_test, y_train, y_test = preprocess_method(df, k=6)
        elif preprocess_name == "Discretization":
            x_train, x_test, y_train, y_test = preprocess_method(df, n_bins=6)
        else:
            x_train, x_test, y_train, y_test = preprocess_method(df)

        for model_name, model_func in [("Decision Tree", hyperparameter_tuning_decision_tree),
                                       ("Naive Bayes", hyperparameter_tuning_naive_bayes),
                                       ("SVC", hyperparameter_tuning_svc)]:
            model_results_df = model_func(x_train, x_test, y_train, y_test)

            model_results_df["Preprocessing Method"] = preprocess_name
            model_results_df["Model"] = model_name
            columns_order = ["Preprocessing Method", "Model", "Accuracy", "Precision", "Recall", "F1 Score", "Parameters"]
            model_results_df = model_results_df[columns_order]
            
            results_df = results_df.append(model_results_df, ignore_index=True)

    results_df = results_df.sort_values(by=['Model', 'Preprocessing Method'])
    results_df.to_csv("results.csv", index=False)


#   ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def main():
    df = read_data()
    data_exploration(df)    
    test_models(df)


if __name__ == '__main__':
    main()
    plt.show()
    