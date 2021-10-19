# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:57:09 2021

@author: luis
"""

# Regresión lineal múltiple en Spyder con Python

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print("LIBRERÍAS IMPORTADAS")

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print("DATATSET IMPORTADO")

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) # El "3" es el número de la columna que se desea convertir
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
#X = onehotencoder.fit_transform(X), Línea de código original, con está NO se modifica el type a float64
X = np.array(onehotencoder.fit_transform(X), dtype=np.float64)
print("DATOS CODIFICADOS X EN TYPE FLOAT64")

## Sentencia para cambiar a type = float64
#ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough')
#X = np.array(ct.fit_transform(X), dtype=np.float64)
#print("CONVERTIDO A TYPE = FLOAT64")

## CÓDIFICAR LA CATEGORÍA Y
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# Evitar la trampa de las variables ficticias
X = X[:, 1:] # Se elimina la una de las variables dummy
print("VARIABLES FICTICIAS ELIMINADA")

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("DATASET DIVIDIDO PARA TRAINING Y TESTING")

# Escalado de variables
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

# Ajustar el modelo de Regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
print("MODELO DE REGRESIÓN READY!!!")

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)
print("PREDICCIÓN DE RESULTADOS DEL DATASET DE TESTING GENERADO")

# NUEVO COMPARACIÓN DE RESULTADOS ENTRE Y_TEST & Y_PREDIC
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(335)
print("TABLA DE COMPARACIÓN INDEXADA AL ENTORNO")

# NUEVO MÉTRICAS DE EVALUACIÓN DEL MODELO R2
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2_score:', metrics.r2_score(y_test, y_pred))

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) # axis = 1 (columna) axis = 0 (fila)
SL = 0.05
print("MODELO OPTIMIZADO")

# Se ha añadido el modificador .tolist() al X_opt para adaptarse a Python 3.7
# Se elimina la variable que tenga el valor P más alto mayor a 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

print('R2_score:', metrics.r2_score(y_test, y_pred))

# Eliminación automática hacia atrás considerando solamente p-valores
import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#Eliminación hacia atrás considerando p-valores y el valor de  R Cuadrado Ajustado

import statsmodels.api as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
