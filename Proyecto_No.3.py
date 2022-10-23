# Proyecto No.3, Regresion Lineal y Modelos de Poblacion
# Grupo No.:
#      Jose Block, 18935
#      Abril Palencia, 18198
# Fecha: 06/10/2022

import math
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# CASO 2
data = pd.read_csv("./data.csv")
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
filtered_data = data[(data['smoothness_mean'] <= 1800 ) & (data['perimeter_mean'] <= 130) & (data['perimeter_mean'] >= 50)&(data['area_mean'] >= 160)]
colores=['#FFB4A4','#7EB86D']
tamanios=[30,60]
 
f1 = filtered_data['area_mean'].values
f2 = filtered_data['perimeter_mean'].values
 
# Diferenciar si el tumor es Benigno o Maligno
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['diagnosis']==1):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])

dataX =filtered_data[["area_mean"]]
X_train = np.array(dataX)
y_train = filtered_data['perimeter_mean'].values
 
# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()
 
# Entrenamos nuestro modelo
regr.fit(X_train, y_train)
y_pred = regr.predict(X_train)

print('Coeficiente: ', regr.coef_)
print('valor donde corta el eje Y (en X=0): ', regr.intercept_)
print("Error medio cuadrado: %.2f" % mean_squared_error(y_train, y_pred))
print('Puntaje de Varianza: %.2f' % r2_score(y_train, y_pred))

# La grafica
plt.scatter(X_train[:,0], y_train,  c=asignar, s=tamanios[0])
plt.plot(X_train[:,0], y_pred, color='#52C8FF', linewidth=2)

plt.xlabel('Área')
plt.ylabel('texture_mean')
plt.title('Regresión Lineal')

plt.show()

# CASO 1

# Importamos los dataset que tiene de ejemplo la libreria
from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_boston()

print(dataset.keys())
print(dataset.feature_names)
print(dataset.DESCR)

x = dataset.data[:, np.newaxis, 5]
y = dataset.target
X_train, X_test, y_train, y_test= train_test_split (x, y, test_size=0.5)

# Creamos el objeto de Regresión Linear
lr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
lr.fit (X_train, y_train)
Y_pred = lr.predict (X_test)

print('Coeficiente: ', lr.coef_)
print('valor donde corta el eje Y (en X=0): ', lr.intercept_)
print("Error medio cuadrado: ", mean_squared_error(X_train, y_train))
print('Puntaje de Varianza:', lr.score(X_train, y_train))

plt.scatter(X_test, y_test)
plt.plot (X_test,Y_pred, color='blue', linewidth=3)
plt.title ('Regresión Lineal')
plt.xlabel ('Número de habitaciones')
plt.ylabel("Valor medio")
plt.show ()

# CASO 3
dataset = pd.read_csv('./Admission_Predict_Ver1.1.csv')
X = dataset.iloc[:len(dataset), 1].values
X = X.reshape(-1,1)
X = np.insert(X, 0, 1, axis = 1)
y = dataset.iloc[:len(dataset), -1].values.reshape(-1,1)

# Seleccionar conjunto de training y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Escalado de las variables
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.transform(X_test)
st_y = StandardScaler()
y_train = st_y.fit_transform(y_train).reshape(-1)
y_test = st_y.transform(y_test).reshape(-1)

regression_py = LinearRegression() 

# Entrenamos nuestro modelo
regression_py.fit(X_train, y_train)
y_predict_py = regression_py.predict(X_test) 

print('Coeficiente: ', regression_py.coef_)
print('valor donde corta el eje Y (en X=0): ', regression_py.intercept_)

# Gráfica 
plt.scatter(X_test[:,1], y_test, color = "#52C8FF") 
plt.plot(X_test[:,1], y_predict_py, color = "blue")
plt.title("Regresión lineal")
plt.xlabel("GRE Score")
plt.ylabel("Probabilidad admisión a Postgrado")
plt.show()