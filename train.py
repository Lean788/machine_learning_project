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

# Leyendo train para el entrenamiento

df = read_csv_from_zip('./src/data.zip', 'train.csv')

print(df.head())

# ==============================================================================

# Dividimos train en features y target.



# ==============================================================================


# ==============================================================================


# ==============================================================================


# ==============================================================================


# ==============================================================================


# ==============================================================================


# ==============================================================================
