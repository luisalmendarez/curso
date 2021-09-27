# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:49:16 2021

@author: luis
"""

# Plantilla de pre-procesado #

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset a Spyder
dataset = pd.read_csv('Data.csv') # Definir la ubicación del archivo
X = dataset.iloc[:, :-1].values # agregar variables independientes
y = dataset.iloc[:, 3].values # agregar variables dependientes (a predecir)

# Conversión de los NA´s o null
from sklearn.impute import SimpleImputer
## strategy = por cuál valor se va a remplazar, mean = media o puedeser medium
## verbose = 0 es igual a media de la columna y 1 es media de la fila
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose=0)
# Seleccionar las columnas en donde se van a reemplazar los NA´s y null
imputer = imputer.fit(X[:,1:3]) 
X[:, 1:3] = imputer.transform(X[:,1:3])

# Codificar los datos categóricos (conversión de valores alfanúmericos a númericos)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
## Proceso de codificación de a variables dummy
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X), dtype=np.float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Dividir el dataset una parte para entrenamiento y otra para testing
from sklearn.model_selection import train_test_split
## Se dividen las variables, por cada train debe haber una test
## El parámetro test_size es para definir el tamaño en porcentaje para testing
## El parámetro random_state es el número de veces que deseas reproducir el algoritmo
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

# Escalado de variables
## Para poder dar un valor equilibrado a cada varible sin importar si hay mucha
## entre cada una de ellas.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)