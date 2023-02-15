# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import missingno as mss
import re

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
import seaborn as sns
style.use('ggplot') or plt.style.use('ggplot')

# Análisis de los textos
# ==============================================================================
from textblob import TextBlob

# Preprocesado y modelado
# ==============================================================================
# from sklearn.decomposition import PCA
# from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats.stats import pearsonr, skew
from scipy.stats import shapiro
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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

# ACCEDIENDO A DF_TEST

test = read_csv_from_zip('./src/src/data.zip', 'test.csv')

# ==============================================================================

# Importamos el modelo escogido
import datetime

now = datetime.datetime.now()
formatted_date = now.strftime("%y%m%d%H%M%S")
model_pattern = f"model_\d{{12}}.pkl"



best_model = load_model_pattern('./src/src/production.zip', model_pattern)


# ==============================================================================

# Dividimos train en features y target.

y_test = test['reviews.rating_binned']
X_test = test.drop(columns='reviews.rating_binned')

# print(X_test.shape)
# print(y_test.shape)

print('\n','\n','-------------- RANDOM FOREST REGRESSOR --------------','\n','\n')

# Escalado

scaler = StandardScaler()

scaler.fit(X_test)

X_test_scaled = scaler.transform(X_test)

# Realizar predicciones en los datos de prueba escalados
y_pred = best_model.predict(X_test_scaled)

print('\n','\n','-------------- RESULTADOS TOTALES DE LAS MÉTRICAS --------------','\n','\n')

print("Score del modelo (R^2):", round(best_model.score(X_test, y_test), 4))
print("R^2 score:", round(r2_score(y_pred, y_test), 4))
print("MAE score:", round(mean_absolute_error(y_pred, y_test), 4))
print("MSE score:", round(mean_squared_error(y_pred, y_test), 4))
print("RMSE score:", round(np.sqrt(mean_squared_error(y_pred, y_test)), 4))

def mean_absolute_percentage_error(y_pred, y_true): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print("MAPE score:", round(mean_absolute_percentage_error(y_pred, y_test), 4))


# ==============================================================================

print('\n','\n', ' Procedemos a guardar la predicción en production:', '\n')

# ==============================================================================


test['prediction'] = y_pred

predictions = pd.DataFrame({'id':test.index, 'predictions':test['prediction'] })

predictions.to_csv('./src/src/prediction/prediction.csv', index = False, encoding='utf-8')

print('Predicción guardada con exito.')
# ==============================================================================

