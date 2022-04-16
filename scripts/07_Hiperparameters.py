# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
import pandas as pd
import numpy as np
from datetime import datetime


def print_result(model, model_trained):
    means = model_trained.cv_results_['mean_test_score']
    stds = model_trained.cv_results_['std_test_score']
    params = model_trained.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(" %s: %f (%f): %r" % (model, mean, stdev, param))


path_DB = 'documentos\\DB_FUT\\'
path_results = 'documentos\\RESULTADOS\\'

df = pd.read_excel(path_DB + 'DadosTreinamento.xlsx')

df = df[df['ano'] < 2020]

df = df[[
    'winner',
    'm_GolsFeitos_mdmv_10',
    'm_GolsSofridos_mdmv_10',
    'm_Faltas_mdmv_10',
    'm_Escanteios_mdmv_10',
    'm_Cruzamentos_mdmv_10',
    'm_QtdDefesas_mdmv_10',
    'm_Impedimentos_mdmv_10',
    'm_posse_mdmv_10',
    'm_chutesaogol_mdmv_10',
    'm_score_mdmv_10',
    'v_GolsFeitos_mdmv_10',
    'v_GolsSofridos_mdmv_10',
    'v_Faltas_mdmv_10',
    'v_Escanteios_mdmv_10',
    'v_Cruzamentos_mdmv_10',
    'v_QtdDefesas_mdmv_10',
    'v_Impedimentos_mdmv_10',
    'v_posse_mdmv_10',
    'v_chutesaogol_mdmv_10',
    'v_score_mdmv_10',
    'v_MdPosse',
    'v_QtdGols',
    'v_QtdAssistencias',
    'v_MdGolsPorPartida',
    'v_MdAssistenciasPorPartida',
    'm_MdPosse',
    'm_QtdGols',
    'm_QtdAssistencias',
    'm_MdGolsPorPartida',
    'm_MdAssistenciasPorPartida'
]]

array = df.values
bos_y = array[:, 0]
bos_X = array[:, 1:].astype(float)
test_size = 0.20
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(bos_X,
                                                                    bos_y,
                                                                    test_size=test_size,
                                                                    random_state=seed)

with open(path_results + "07_Resultados.txt", "a") as stream:
    # Parâmetros
    num_folds = 10
    scoring = 'accuracy'
    # definindo uma semente global
    np.random.seed(7)

    print("Execucao: " + str(datetime.now()), file=stream)
    print("num_folds: " + str(num_folds), file=stream)
    print("scoring: " + str(scoring), file=stream)

    # Tuning do LogisticRegression
    rescaledX = X_train
    C = np.logspace(-4, 4, 10)
    penalty = ['l2']
    param_grid = dict(penalty=penalty, C=C)
    model = LogisticRegression(solver='newton-cg')
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)
    print("Melhor LR: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_))
    print("Melhor LR: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_), file=stream)
    #print_result("LR", grid_result)

    # Tuning do KNeighborsClassifier
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    n_neighbors = [5, 10, 15, 20, 30, 40, 50]
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan']
    param_grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
    model = KNeighborsClassifier()
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)
    print("Melhor KNN: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_))
    print("Melhor KNN: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_), file=stream)
    #print_result("KNN", grid_result)

    # Tuning do TreeClassifier
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    # Parameters
    max_depth = [5, 10, 15, 20, 30, 40, 50, 70,
                 75, 80, 85, 90, 95, 100, 110, 120, 150]
    criterion = ['gini', 'entropy']
    param_grid = dict(max_depth=max_depth, criterion=criterion)
    # Model configurations
    model = DecisionTreeClassifier()
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)
    print("Melhor CART: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_))
    print("Melhor CART: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_), file=stream)
    #print_result("CART", grid_result)
    # Tuning do GaussianNB
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    var_smoothing = np.logspace(0, -9, num=100)
    param_grid = dict(var_smoothing=var_smoothing)
    model = GaussianNB()
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)
    print("Melhor NB: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_))
    print("Melhor NB: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_), file=stream)
    #print_result("NB", grid_result)

    # Tuning do SVM
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    c_values = [0.1, 0.5, 1.0, 1.5, 2.0]
    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
    param_grid = dict(C=c_values, kernel=kernel_values)
    model = SVC()
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)
    print("Melhor SVM: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_))
    print("Melhor SVM: %f com %s" %
          (grid_result.best_score_, grid_result.best_params_), file=stream)
    #print_result("SVM", grid_result)

    print("----------------------------------------------------\n", file=stream)
