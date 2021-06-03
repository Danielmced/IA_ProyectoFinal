# IA_ProyectoFinal
Proyecto final para Inteligencia Artificial, Universidad 
Mariano Galvez de Guatemala

# Integrantes
```python
# Carlos José Abraham Bethancourt Zúñiga 1290-17-7166
# Daniel Estuardo Cabrera Misa 1290-17-11799
# Pablo Andres Bonilla Morales 1290-17-10498
# Evelyn Yolanda Zamora Hernández 1290-17-5541
```

# Ponlo en marcha
Debes tener el creditcard.csv al mismo nivel de carpetas que el proyecto ProyectoFinal.py, por 
motivos de espacio este documento ha sido adjuntado en Blackboard.

luego puedes ejecutar:
```Bash
python ProyectoFinal.py
```

en caso que tengas disponible python3:

```Bash
python3 ProyectoFinal.py
```

# Qué realiza el proyecto

Sistema de aprendizaje profundo el cual por medio de un modelo de Keras secuencial analiza por 
medio de tres redes neuronales densas y dos redes neuronales dropout los datos de una bataset 
asociada para la comprobación de la legibilidad de una tarjeta de crédito.

# Librerias, Modulos y Frameworks utilizados:

## CSV

[CSV File Reading and Writing](https://docs.python.org/3/library/csv.html)

El módulo csv implementa clases para leer y escribir datos tabulares en formato CSV. Permite a los 
programadores escribir o leer, sin conocer los detalles precisos del formato CSV utilizado por Excel. 
También permite describir formatos CSV entendidos por otras aplicaciones o definir sus propios 
formatos CSV para propósitos especiales.

## NumPy

[NumPy v1.20.0](https://numpy.org/install/)

Es una biblioteca que da soporte para crear vectores y matrices grandes multidimensionales, junto 
con una gran colección de funciones matemáticas de alto nivel para operar con ellas.

## SMOTE

[SMOTE from imblearn.over_sampling](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

Es una técnica de sobremuestreo en la que las muestras sintéticas se generan para la clase minoritarial, 
este algoritmo ayuda a superar el problema de sobreajuste que plantea el sobremuestreo aleatorio. 
Se centra en el espacio de características para generar nuevas instancias con la ayuda de la interpolación 
entre las 
instancias positivas que se encuentran juntas.

## Train_test_split

[Train_test_split from sklearn.model_selection](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

Es un método para establecer un plan para analizar datos y luego usarlo para medir nuevos datos. Seleccionar 
un modelo adecuado le permite generar resultados precisos al hacer una predicción. Para hacer eso, necesitas 
entrenar tu modelo usando un conjunto de datos específico.

## Keras

[Keras from tensorflow](https://keras.io/)

Es una biblioteca de redes neuronales de código abierto escrita en python. Esta especialmente diseñada para 
posibilitar la experimentación en más o menos poco tiempo con redes de aprendizaje profundo. Sus fuertes se 
centran en ser amigable para el usuario, modular y extensible.

## Keras Backend

[Backend from keras](https://keras.rstudio.com/articles/backend.html)

Keras es una biblioteca a nivel de modelo que proporciona bloques de construcción de alto nivel para desarrollar 
modelos de aprendizaje profundo.Pero no maneja por sí mismo operaciones de bajo nivel como productos tensoriales, 
convoluciones, etc. En cambio, se basa en una biblioteca de manipulación de tensores especializada y bien 
optimizada para hacerlo, que actúa como el "backend engine" de Keras. 

