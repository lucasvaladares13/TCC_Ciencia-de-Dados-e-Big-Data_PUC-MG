# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import (model_selection)
import pandas as pd
import numpy as np
from datetime import datetime

path_DB = 'documentos\\DB_FUT\\'
path_images = 'images\\'
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

# Parâmetros
num_folds = 10
scoring = 'accuracy'
# definindo uma semente global
np.random.seed(7)
# Criação dos modelos
models = []
models.append(('LR', LogisticRegression(solver='newton-cg')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Avaliação dos modelos

results = []
names = []
with open(path_results + "06_Resultados.txt", "a") as stream:
    print("Execucao: " + str(datetime.now()), file=stream)
    print("Treinando modelos com dados normais")
    print("Treinando modelos com dados normais", file=stream)
    for name, model in models:
        kfold = KFold(n_splits=num_folds)
        cv_results = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        print(msg, file=stream)

    # Comparação dos modelos
    fig = plt.figure()
    fig.suptitle("Comparação dos Modelos")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig(path_images + "06_" + "BoxPlot_Models" + '.png', dpi=300)
    # plt.show()
    plt.close()

    # Padronização do dataset
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline(
        [('Scaler', StandardScaler()), ('LR', LogisticRegression(solver='newton-cg'))])))
    pipelines.append(('ScaledKNN', Pipeline(
        [('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))
    pipelines.append(('ScaledCART', Pipeline(
        [('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
    pipelines.append(('ScaledNB', Pipeline(
        [('Scaler', StandardScaler()), ('NB', GaussianNB())])))
    pipelines.append(('ScaledSVM', Pipeline(
        [('Scaler', StandardScaler()), ('SVM', SVC())])))
    results = []
    names = []
    print("Treinando modelos com dados normalizados")
    print("Treinando modelos com dados normalizados", file=stream)
    for name, model in pipelines:
        kfold = KFold(n_splits=num_folds)
        cv_results = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        print(msg, file=stream)

    print("----------------------------------------------------\n", file=stream)
    # Comparação dos modelos
    fig = plt.figure()
    fig.suptitle("Comparação dos Modelos normalizados")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig(path_images + "06_" +
                "BoxPlot_Models_Normalizados" + '.png', dpi=300)
    # plt.show()
    plt.close()
