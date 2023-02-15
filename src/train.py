# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import missingno as mss
import re
import os

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
import seaborn as sns
style.use('ggplot') or plt.style.use('ggplot')

# Análisis de los textos
# ==============================================================================
# from textblob import TextBlob

# Preprocesado y modelado
# ==============================================================================
# from sklearn.decomposition import PCA
# from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import roc_curve
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectKBest, f_regression
# from scipy.stats.stats import pearsonr, skew
# from scipy.stats import shapiro
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import scale

# Configuración importación de funciones
# ==============================================================================
import sys
sys.path.append('./utils/')

from utils.functions import save_model

# Importación de funciones
# ==============================================================================
from utils.functions import *

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Configuración visualización de data
# ==============================================================================
pd.options.display.max_rows = 35
pd.options.display.max_columns = 35
pd.options.display.max_colwidth = 50
# ==============================================================================

# Leyendo train para el entrenamiento

train_df = read_csv_from_zip('./src/src/data.zip', 'train.csv')


# ==============================================================================

# Dividimos train en features y target.

y = train_df['reviews.rating_binned']
X = train_df.drop(columns='reviews.rating_binned')

# print(X.shape)
# print(y.shape)

# Dividimos train y test.

seed = 10

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=seed)

# print(X_train.shape) 
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# ==============================================================================

# Metemos en un pipeline el escalado 
pipe_escal = Pipeline([
    ('scaler', StandardScaler())
])

X_train_scaled = pipe_escal.fit_transform(X_train)
X_test_scaled = pipe_escal.transform(X_test)


# ==============================================================================

#HIPERPARÁMETROS

# RANDOM FOREST REGRESSOR
grid_rf_reg = { "n_estimators": [190, 220, 250, 300, 350], # Número de árboles en el bosque.
"rfr__max_depth": [9, 10, 12, 15, 18], # Profundidad máxima de los árboles.
"rfr__min_samples_split": [2, 5, 10], # Número mínimo de samples requeridos para hacer un split en un nodo
"rfr__min_samples_leaf": [1, 2, 4], # Número mínimo de samples en un nodo
"rfr__max_features": ["auto", "sqrt", "log2"] # Número de features a considerar en cada split
}

# ==============================================================================

# DEFINIMOS PIPELINES

pipeline_rfr = Pipeline([
("scaler", StandardScaler()),
("rfr", RandomForestRegressor())
])

# ==============================================================================

grid_random_forest_reg = {
"rfr__n_estimators": [190, 220, 250, 300, 350],
"rfr__max_depth": [9, 10, 12, 15, 18],
"rfr__max_features": ["auto", "sqrt", "log2"],
}

# ==============================================================================

# ENTRENAMIENTO


# Random Forest Regression GridSearchCV

grid_search_rfr = GridSearchCV(pipeline_rfr, grid_random_forest_reg, cv=10, scoring="neg_mean_squared_error", verbose=1, n_jobs=-1)
grid_search_rfr.fit(X_train, y_train)

# ==============================================================================

print('\n','\n','\n','-------------- R E S U L T A D O S --------------','\n','\n','\n')


# ==============================================================================
print('\n','\n','-------------- RANDOM FOREST REGRESSOR --------------','\n','\n')

# ==============================================================================

# Almacenamos el nombre del modelo y su mejor score en una lista de tuplas
best_grid_rfr = ('Random Forest Regressor 2nd chance', grid_search_rfr.best_score_)

# Convertimos la lista de tuplas en un DataFrame
best_grids = [best_grid_rfr]
best_grids_df_rfr = pd.DataFrame(best_grids, columns=['Model', 'Best Score'])

# Ordenamos el DataFrame por el mejor score en orden descendente
best_grids_df_rfr.sort_values(by='Best Score', ascending=False, inplace=True)

# Mostramos el resultado
print(best_grids_df_rfr)

# ==============================================================================
print('\n','\n', ' El mejor estimador ha sido:', '\n')
# ==============================================================================

# El mejor ha sido RFR. Ya esta entrenada con todo train
print(grid_search_rfr.best_estimator_)

# ==============================================================================

# La probamos el mejor estimador del modelo
print ('\n','Probamos el best_estimator: ',grid_search_rfr.best_estimator_.score(X_test, y_test))

# ==============================================================================

print('\n','\n', ' Procedemos a guardar los resultados en modelos entrenados:', '\n')

# ==============================================================================

# Guardar los modelos

save_model(grid_search_rfr, zip_file="./src/src/production.zip")

print('Modelo guardado con exito.')
# ==============================================================================
