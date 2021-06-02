#Integrantes
#Carlos José Abraham Bethancourt Zúñiga 1290-17-7166
#Daniel Estuardo Cabrera Misa 1290-17-11799
#Pablo Andres Bonilla Morales 1290-17-10498
#Evelyn Yolanda Zamora Hernández 1290-17-5541

import csv
import numpy as np

#lectura y balanceo de datos....................................................................................................................................................................................
#Nombre del fichero que quiero leer
datos = "creditcard.csv"

#Inicialización de los arrays
caracteristicas = []
clase = []

#Lectura de datos
with open(datos) as f:
    for i, linea in enumerate(f):
        #Saltamos la cabecera del csv
        if i == 0:
            print("Cabecera:", linea.strip())
            continue  
        #Introducimos los datos en los arrays de características
        campos = linea.strip().split(",")
        caracteristicas.append([float(v.replace('"', "")) for v in campos[:-1]])
        clase.append([int(campos[-1].replace('"', ""))])
        if i == 1:
            print("Ejemplo de caracteristicas:", caracteristicas[-1])

#Conversión de datos
data = np.array(caracteristicas, dtype="float32")
target = np.array(clase, dtype="uint8")

#Pintamos la forma de los datos
print("Forma de los datos de entrada al modelo:", data.shape)
print("Forma de las clases de salida:", target.shape)

#BALANCEO
#Primero analizamos los datos

#Inicializacion de la cuenta
legal = 0
fraude = 0

#Cuenta de datos
for x in range(target.shape[0]):
    if target[x] == 0:
        legal = legal + 1
    else:
        fraude = fraude + 1

#Representación
print("Tarjetas legales: " + str(legal))
print("Tarjetas fraudulentas: " + str(fraude))

#Sobremuestreamos por la gran diferencia

#Importamos los paquetes de sobremuestreo
from imblearn.over_sampling import SMOTE

#SMOTE
smote = SMOTE()

#Generación de nuevas muestras sintéticas
dataSmote, targetSmote = smote.fit_resample(data,target)

#Volvemos a contar
legal = 0
fraude = 0

#Cuenta de datos
for x in range(targetSmote.shape[0]):
    if targetSmote[x] == 0:
        legal = legal + 1
    else:
        fraude = fraude + 1

#Representación
print("Tarjetas legales balanceadas: " + str(legal))
print("Tarjetas fraudulentas balanceadas: " + str(fraude))

#División de datos en conjunto de evaluación y conjunto de entrenamiento
from sklearn.model_selection import train_test_split
dataTrain, dataTest, targetTrain, targetTest = train_test_split(dataSmote,targetSmote, random_state = 0)

#Normalización de los datos 

#Cálculo de la media
mean = np.mean(dataTrain, axis=0)

#Restamos a las características la media
dataTrain -= mean
dataTest -= mean

#Cálculo de la desviación estándar
std = np.std(dataTrain, axis=0)

#Dividimos entre la desviación estándar
dataTrain /= std
dataTest /= std

#Construcción del modelo con Keras...........................................................................................................................................................................

#Añadimos las capas de nuestra red neuronal (3 densas y dos de dropout)
from tensorflow import keras
model = keras.Sequential(
    [
        #Capa densa, la primera capa siempre tiene que especificar la forma de entrada
        keras.layers.Dense(
            256, activation="relu", input_shape=(dataTrain.shape[-1],) #Nodos de la capa densa, y función de activación
        ),
        keras.layers.Dense(256, activation="relu"),
        #Capa de Dropout. Inactiva algunos de los nodos de la red para evitar el sobreentrenamiento
        keras.layers.Dropout(0.3), #El atributo que se pone es el ratio de inactivación
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

#Vemos la forma de nuestro modelo
model.summary()

#Funciones de las métricas
from keras import backend as K

#Funciones de las métricas
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Compilamos el modelo
model.compile(
    optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=['accuracy',f1_m,precision_m, recall_m]
) #Compilamos el modelo con su optimizador, la forma en la que actualizamos los pesos (minimización) y en base a qué métricas

#Lo entrenamos con los datos de entrenamiento
model.fit(
    dataTrain,
    targetTrain,
    batch_size=2048,
    epochs=10,
    verbose=1,
    validation_data=(dataTest, targetTest),
)#Conjuntos de entrenamientos y evaluación, numero de muestras en la propagación hacia atrás, 
#numero de iteraciones para mejorar el modelo, la verbosidad y los conjuntos de validación

#Métricas
#Conjunto de evaluación
print()
print("Datos sobre la evaluación")
loss, accuracy,f1_score, precision, recall = model.evaluate(dataTest, targetTest, verbose=False)
print("Exactitud de evaluación: {:.4f}".format(accuracy))
print("F1 de evaluación: {:.4f}".format(f1_score))
print("Precisión de evaluación: {:.4f}".format(precision))
print("Memoria de evaluación: {:.4f}".format(recall))