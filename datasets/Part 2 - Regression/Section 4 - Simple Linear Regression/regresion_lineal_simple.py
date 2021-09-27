# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:47:50 2021

@author: luis
"""


# Regresión lineal simple
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset a Spyder
dataset = pd.read_csv('Salary_Data.csv') # Definir la ubicación del archivo
X = dataset.iloc[:, :-1].values # agregar variables independientes
y = dataset.iloc[:, 1].values # agregar variables dependientes (a predecir)

# Dividir el dataset una parte para entrenamiento y otra para testing
from sklearn.model_selection import train_test_split
"""Se dividen las variables, por cada train debe haber una test
El parámetro test_size es para definir el tamaño en porcentaje para testing
El parámetro random_state es el número de veces que deseas reproducir el algoritmo"""
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 1/3, random_state = 0)

# Escalado de variables
## Para poder dar un valor equilibrado a cada varible sin importar si hay mucha
## entre cada una de ellas.
"""EN LA REGRESIÓN LINEAL SIMPLE NO REQUIERE ESCALADO"""
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Crear el modelo de Regresión Lineal con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression() # Para invocar la librería
regression.fit(X_train, y_train) # modelo fit, deben contener el mismo número de filas y columnas

# Cómo predecir el conjunto de test
y_pred = regression.predict(X_test) #X_test = el dataset que deseas predecir

#Gráficar los resultados del entrenamiento
plt.scatter(X_train, y_train, color = "red") # Gráfica de dispersión color rojo
plt.plot(X_train, regression.predict(X_train), color = "blue") # Recta de regresión
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)") #Título de la gráfica
plt.xlabel("Años de Experiencia") # Título del eje X
plt.ylabel("Sueldo (en $)") # Título del eje y
plt.show() # Sentencia para imprimir todo

#Gráficar los resultados del testing
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()