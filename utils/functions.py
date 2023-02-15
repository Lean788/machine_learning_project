import numpy as np
import pandas as pd
import statsmodels.api as sm
import missingno as mss
import re
from textblob import TextBlob
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os




# Función que recoge esta transformación de reviews.date

def extract_date(df, date_column_name):
    '''
    Función para extraer de la columna reviews.date la fecha en dicho formato.
    Además se crean las nuevas columnas Year y Month
    '''
    df[date_column_name] = df[date_column_name].str.replace(r'T.*$', '')
    df[date_column_name] = pd.to_datetime(df[date_column_name])
    df['Year'] = df[date_column_name].dt.year
    df['Month'] = df[date_column_name].dt.month
    return df


# Definimos la función que añada este dato externo y lo una al data

def add_extern_data_crime(df, path_ext_data='./data/state_crime.csv'):

    '''
    Función para cargar los datos de crime.csv,
    mapear y renombrar los estados por sus acrónimos y
    finalmente, unir al dataframe general
    '''

    # Importar los datos externos
    crime = pd.read_csv(path_ext_data)
    
    # Seleccionar las columnas relevantes
    crime = crime[['State', 'Year', 'Data.Population','Data.Totals.Violent.All']]
    
    # Mapear los nombres de estado a su código de 2 letras
    states = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
              'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
              'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
              'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
              'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
              'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
              'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
              'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
              'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
              'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}
    crime['State'] = crime['State'].replace(states)
    
    # Renombrar las columnas
    crime.rename(columns={"State": "province", "Data.Population": "population", "Data.Totals.Violent.All": "totals_crimes"}, inplace=True)
    
    # Unir los datos externos al DataFrame principal
    df = pd.merge(df, crime, on=['province', 'Year'])
    
    return df



# Definimos la función que añada este dato externo y lo una al df

def add_extern_df_distance(df, path_ext_df='./data/center_province.csv'):

    '''
    Función para cargar los datos de crime.csv,
    mapear y renombrar los estados por sus acrónimos y
    finalmente, unir al dfframe general
    '''

    # Importar los datos externos
    center_province = pd.read_csv(path_ext_df, sep=';')

    # Seleccionar las columnas relevantes
    center_province = center_province.iloc[:,1:]
    
    # Mapear los nombres de estado a su código de 2 letras
    states = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
              'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
              'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
              'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
              'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
              'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
              'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
              'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
              'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
              'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}
    
    center_province['name'] = center_province['name'].replace(states)
    
    # Renombrar las columnas
    center_province.rename(columns={"latitude": "latitude_center", 'longitude': 'longitude_center', 'name': 'province'}, inplace=True)
    
    # Unir los datos externos al dataFrame principal
    df = pd.merge(df, center_province, on=['province'])

    # Convertimos a float estas columnas
    df['latitude_center'] = df['latitude_center'].astype('float')
    df['longitude_center'] = df['longitude_center'].astype('float')
    
    return df


# función para calcular la distancia en metros
def haversine_distance(latitude, longitude, latitude_center, longitude_center):
    from math import radians, sin, cos, sqrt, atan2

    '''
    La fórmula de Haversine es una fórmula que permite calcular la distancia entre dos puntos en una esfera a partir de sus latitudes y longitudes.
    '''

    R = 6371e3 # radio de la Tierra en metros
    phi1 = radians(latitude)
    phi2 = radians(latitude_center)
    delta_phi = radians(latitude_center-latitude)
    delta_lambda = radians(longitude_center-longitude)
    a = sin(delta_phi/2)**2 + cos(phi1)*cos(phi2)*sin(delta_lambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance



# ANÁLISIS DE SENTIMIENTO DE TEXTO

def clean_text(text):
    # Eliminar símbolos y números
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    
    # Convertir a minúsculas
    text = text.lower()
    
    return text

def count_words(df, column):
    df['wordCount'] = column.apply(lambda x: len(x.split()))

def sentiment_analysis(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.10:
        return "Positive"
    elif analysis.sentiment.polarity < -0.10:
        return "Negative"
    else:
        return "Neutral"
    
def value_polarity(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return polarity

def value_subjectivity(text):
    analysis = TextBlob(text)
    subjectivity = analysis.sentiment.subjectivity
    return subjectivity



# Función para encerrar todo el proceso de transformación y creación en el análisis de sentimiento:

def sentiment_analysis_full(df, column_text):

    '''
    Creo una nueva columna clean_text que contiene el texto limpio de la columna reviews.text.
    Creo la columna sentiment que contiene el sentimiento del texto en la columna clean_text (positivo, negativo o neutral).
    Creo la columna wordCount que contiene el número de palabras en la columna clean_text.
    Creo la columna polaritySentiment que contiene la polaridad del sentimiento en la columna clean_text.
    Finalmente, creo la columna subjectivitySentiment que contiene la subjetividad del sentimiento en la columna clean_text.
    '''

    # Limpiar texto
    df[column_text + '_clean'] = df[column_text].apply(lambda x: clean_text(x))

    # Análisis de sentimiento
    df[column_text + "_sentiment"] = df[column_text + '_clean'].apply(sentiment_analysis)

    # Contar palabras
    count_words(df, df['clean_text'])

    # Polaridad de sentimiento
    df[column_text + "_polarity"] = df[column_text + '_clean'].apply(value_polarity)

    # Subjetividad de sentimiento
    df[column_text + "_subjectivity"] = df[column_text + '_clean'].apply(value_subjectivity)

    return df



# Generamos la función para encapsular la extracción de la raiz de la web:


def web_homeIndex(df):
    import re
    # Definir la función para extracción de páginas web
    def extract_website(text):
        pattern = re.compile(r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)')
        match = re.search(pattern, text)
        if match:
            return match.group()
        return None
    
    # Agregar columna 'websites_ratings' utilizando la función extract_website
    df['websites_ratings'] = df['sourceURLs'].apply(extract_website)
    
    return df


# Función para eliminar las columnas seleccionadas:

def del_null_values(df, columns_to_delete=['reviews.userCity', 'reviews.userProvince']):
    df.drop(columns=columns_to_delete, inplace=True)
    return df



# encapsulamos en función, tanto para reviews.text_polarity como para reviews.text_:

def corregir_outliers(df, start_year, end_year, column):
    # Filtrar el datafame para incluir solo los años especificados
    data_filtered = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]

    # Calcular los valores de los cuartiles superior e inferior
    Q1 = data_filtered[column].quantile(0.25)
    Q3 = data_filtered[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Corregir valores atípicos utilizando .loc
    df.loc[(df["Year"] >= start_year) & (df["Year"] <= end_year) & (df[column] < lower_bound), column] = lower_bound
    df.loc[(df["Year"] >= start_year) & (df["Year"] <= end_year) & (df[column] > upper_bound), column] = upper_bound

    return df


# Una vez hecho las comprobaciones, creamos la función: 

def binning(df, target_col, bins= [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], labels= [1, 2, 3, 4, 5]):
    """
    Aplica la técnica de binning a una columna de un DataFrame.

    Parameters:
    df (DataFrame): El DataFrame que contiene la columna a binar.
    target_col (str): La columna a binar.
    bins (list): La lista de bins para discretizar los valores, que dejaremos por defecto los que necesitamos para este problema.
    labels (list): Las etiquetas para los bins, que dejaremos por defecto los que necesitamos para este problema.

    Returns:
    pandas DataFrame: El DataFrame original con una nueva columna que representa la columna objetivo binada.
    """
    df[target_col + "_binned"] = pd.cut(df[target_col], bins, labels=labels, right=False).astype('int')
    return df


# Una vez hecho las comprobaciones, creamos la función y se lo aplicamos a data.

def oversample_data(X, y):
    print(Counter(y))
    # define oversampling strategy
    sm = SMOTE(sampling_strategy='auto')
    # fit and apply the transform
    X_over,y_over = sm.fit_resample(X,y)
    # summarize class distribution
    print(Counter(y_over))
    
    return X_over, y_over


# Encapsulamos en una función
def escaler_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scal = scaler.transform(X_train)
    X_test_scal = scaler.transform(X_test)
    return X_train_scal, X_test_scal


def save_model(model, file_path):


    with open(file_path, "wb") as archivo_salida:
        pickle.dump(model, archivo_salida)


def load_pipeline(filepath):
    with open(filepath, "rb") as archivo_entrada:
        pipeline_importada = pickle.load(archivo_entrada)
    return pipeline_importada