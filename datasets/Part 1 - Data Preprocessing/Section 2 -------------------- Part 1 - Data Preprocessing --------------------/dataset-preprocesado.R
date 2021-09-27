# Plantilla el pre-procesado de datos
# Importar el dataset
datasets = read.csv("Data.csv")
#datasets = datasets[,2:3] cuando solo se ocupan columnas del datset

# Tratamiento de los NA´s y null
datasets$Age = ifelse(is.na(datasets$Age),
                      ave(datasets$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                      datasets$Age)
datasets$Salary = ifelse(is.na(datasets$Salary),
                      ave(datasets$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                      datasets$Salary)

# Codificar las variables categóricas
datasets$Country = factor(datasets$Country,
                          levels = c("France", "Spain", "Germany"),
                          labels = c(1, 2, 3)) # Se pueden elegir valores aleatorios
datasets$Purchased = factor(datasets$Purchased,
                            levels = c("No", "Yes"),
                            labels = c(0, 1)) # Se puede elegir valores aleatorios, 0 = No y 1 = Yes

# Cómo dividir el dataset en conjunto de entrenamieto y testing
# install.packages("caTools")
# library(caTools)
set.seed(123) ## Semilla aleatoria o random_state
split = sample.split(datasets$Purchased, SplitRatio = 0.8) # Variable a predecir y el SplitRatio = % a entrenar
training_set = subset(datasets, split == TRUE)
testing_set = subset(datasets, split == FALSE)

# Escalado de valores
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])
