# Regresión lineal simple

# Plantilla el pre-procesado de datos
# Importar el dataset
datasets = read.csv("Salary_Data.csv")
#datasets = datasets[,2:3] cuando solo se ocupan columnas del datset

#PARA ESTE MODELO NO SE OCUPA EL TRATAMIENTO DE NA#
#NI CONVERSIÓN DE DATOS CATEGÓRICOS#

# Cómo dividir el dataset en conjunto de entrenamieto y testing
# install.packages("caTools")
# library(caTools)
set.seed(123)
split = sample.split(datasets$Salary, SplitRatio = 2/3)
training_set = subset(datasets, split == TRUE)
testing_set = subset(datasets, split == FALSE)

# Escalado de valores
#training_set[,2:3] = scale(training_set[,2:3])
#testing_set[,2:3] = scale(testing_set[,2:3])

#Creación del modelo de regresión lineal simple
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predecir resultados con el conjunto de testing
y_pred = predict(regressor, newdata = testing_set)

# Visualización de los resultados en el conjunto de entrenamiento
#install.packages("ggplot2")
#library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)") +
  xlab("Años de Experiencia") +
  ylab("Sueldo (en $)")

# Visualización de los resultados en el conjunto de testing
ggplot() + 
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Testing)") +
  xlab("Años de Experiencia") +
  ylab("Sueldo (en $)")
