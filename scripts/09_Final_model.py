# Imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import (model_selection, preprocessing)
from datetime import datetime


# Parâmetros
num_folds = 10
scoring = 'accuracy'
# definindo uma semente global
seed = np.random.seed(7)

path_DB = 'documentos\\DB_FUT\\'
path_images = 'images\\'
path_results = 'documentos\\RESULTADOS\\'

df = pd.read_excel(path_DB + 'DadosTreinamento.xlsx')

df_valid = df[df['ano'] == 2020].copy()
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

df_valid = df_valid[[
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


array_valid = df_valid.values
bos_y_val = array_valid[:, 0]
bos_X_val = array_valid[:, 1:].astype(float)
test_size_val = 0.90


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(bos_X,
                                                                    bos_y,
                                                                    test_size=test_size,
                                                                    random_state=seed)
X_train_val, X_test_val, Y_train_val, Y_test_val = model_selection.train_test_split(bos_X_val,
                                                                                    bos_y_val,
                                                                                    test_size=test_size_val,
                                                                                    random_state=seed)


# Preparação do modelo
# Criação dos modelos
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler(
)), ('LR', LogisticRegression(solver='newton-cg', C=0.0001, penalty='l2'))])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler(
)), ('KNN', KNeighborsClassifier(metric='manhattan', n_neighbors=30, weights='uniform'))])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler(
)), ('CART', DecisionTreeClassifier(criterion='gini', max_depth=5))])))
pipelines.append(('ScaledNB', Pipeline(
    [('Scaler', StandardScaler()), ('NB', GaussianNB(var_smoothing=1))])))
pipelines.append(('ScaledSVM', Pipeline(
    [('Scaler', StandardScaler()), ('SVM', SVC(C=0.5, kernel='poly'))])))

with open(path_results + "09_Resultados.txt", "a") as stream:

    print("Execucao: " + str(datetime.now()), file=stream)
    print("num_folds: " + str(num_folds), file=stream)
    print("scoring: " + str(scoring), file=stream)
    print("Resultados:\n ", file=stream)

    for name, model in pipelines:
        model.fit(X_train, Y_train)
        # Estimativa da acurácia no conjunto de teste
        predictions = model.predict(X_test)
        predictions_val = model.predict(X_test_val)
        Accuracy_score = accuracy_score(Y_test, predictions)
        Accuracy_score_Valid = accuracy_score(Y_test_val, predictions_val)

        msg = "%s: %f (%f)" % (name, Accuracy_score, Accuracy_score_Valid)
        print(msg)
        print(msg, file=stream)

        # Matriz de confusão
        print("\nMatriz de confusao " + name +
              " - Dados de Teste\n ", file=stream)
        cm = confusion_matrix(Y_test, predictions)
        labels = ["1", "2", "3"]
        cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
        cmd.plot(values_format="d")

        plt.title(name,
                  fontdict={'family': 'serif',
                            'color': 'darkblue',
                            'weight': 'bold',
                            'size': 18})
        plt.savefig(path_images + "09_" + name + '.png', dpi=300)
        # plt.show()
        plt.close()
        print(classification_report(Y_test, predictions,
                                    target_names=labels, zero_division=0))
        print(classification_report(Y_test, predictions,
                                    target_names=labels, zero_division=0), file=stream)

        print("\nMatriz de confusao " + name +
              " - Dados de Validacao\n ", file=stream)
        cm = confusion_matrix(Y_test_val, predictions_val)
        labels = ["1", "2", "3"]
        cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
        cmd.plot(values_format="d")

        plt.title(name + "Validação",
                  fontdict={'family': 'serif',
                            'color': 'darkblue',
                            'weight': 'bold',
                            'size': 18})
        plt.savefig(path_images + "09_" + name + '_Validacao.png', dpi=300)
        # plt.show()
        plt.close()
        print(classification_report(Y_test_val, predictions_val,
                                    target_names=labels, zero_division=0))
        print(classification_report(Y_test_val, predictions_val,
                                    target_names=labels, zero_division=0), file=stream)
    print("------------------------------------------------------------\n", file=stream)
