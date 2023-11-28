import time
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn2pmml import PMMLPipeline, sklearn2pmml, make_pmml_pipeline
from sklearn2pmml.feature_extraction.text import Splitter


# En primer lugar vamos a cargar el archivo "datos_entrenamiento.csv" que contiene los 
# datos de entrenamiento previamente seleccionados para generar el modelo. 
# El formato de este archivo CSV tiene dos columnas, la primera contiene la solicitud (Request) 
# y la segunda columna contiene la clasificación de esa solicitud en particular (Classification). 
# La clasificación puede ser "Attack" o "Valid", si se trata de un ataque o no.
train_file = 'datos_entrenamiento.csv'

# Cargamos el archivo en un DataFrame de pandas
# Nota: los atributos al llamar la funcion son para que interprete correctamente las filas
train_data = pd.read_csv(train_file, header=0, quotechar='"', sep=',', escapechar='\\')

# Separamos en dos listas lo que vamos a considerar como los datos (X) que son las solicitudes y 
# las clasificaciones (Y) para cada instancia de los datos de entrenamiento.
X_train = train_data.iloc[:, 0]
Y_train = train_data.iloc[:, 1]

# Ahora viene la la parte más interesante del asunto, que es la generación del modelo. 
# Vamos a usar la clase PMMLPipeline de SciKit Learn, ya que nos permite exportar el modelo a 
# PMML e integrarlo con otras herramientas, como vimos en la edición anterior.
pipeline = PMMLPipeline([
    # Como segundo paso, aplicamos el modelo TF-IDF para identificar los token que deben tener 
    # mayor peso en el modelo para la evaluación de nuevas instancias. Eso lo hacemos con el 
    # TfidfVectorizer. Los parámetros utilizados hacen que solo se tomen los espacios, tabuladores 
    # y saltos de línea como separadores de token, que se utilicen caracteres especiales como 
    # puntos y comas como tokens y que no se hagan otras modificaciones a los datos.
    ('tfidf', TfidfVectorizer(analyzer="word", preprocessor=None, strip_accents=None,
                              tokenizer=Splitter(), token_pattern=r"(?u)\b\S+\b", stop_words=None,
                              binary=False, use_idf=True, norm=None)),
    # Por último dentro del modelo, se seleccionan los 1000 atributos con mayor peso para la 
    # predicción utilizando SelectKBest().
    ('mutual_info_class', SelectKBest(chi2, k=1000)),
    # Como primer paso de la creación del modelo, tenemos que definir cual es el algoritmo a utilizar, 
    # que en nuestro caso es un Random Forest, por lo que utilizamos el RandomForestClassifier.
    ('cls', RandomForestClassifier()),

])


# Ahora que ya tenemos preparados los datos de entrenamiento y la configuración del modelo, es 
# momento de generarlo. Esto se hace simplemente con la función fit() del pipeline, pasando los 
# datos de entrenamiento como atributos.
pipeline.fit(X_train, Y_train)

# Cargamos los datos de prueba desde el archivo "datos_prueba.csv" que tiene el mismo formato que 
# el archivo de datos de entrenamiento. De la misma forma, separamos los datos en X (Solicitudes) 
# e Y (Clasificación).
test_file = 'datos_prueba.csv'

# Cargamos el archivo en un DataFrame de pandas
# Nota: los atributos al llamar la funcion son para que interprete correctamente las filas
test_data = pd.read_csv(test_file, header=0, quotechar='"', sep=',', escapechar='\\')

X_test = test_data.iloc[:, 0]
Y_test = test_data.iloc[:, 1]

# Con los datos de prueba preparados, realizamos una predicción, es decir, le pedimos al modelo 
# que a partir de las solicitudes de prueba, genere las predicciones correspondientes. Estas 
# predicciones son las que vamos a comparar con las clasificaciones existentes, para evaluar el 
# desempeño del modelo.
Y_pred = pipeline.predict(X_test)

# Luego de generadas las predicciones, evaluamos el modelo con la matriz de confusión, el reporte 
# de clasificación y el puntaje de precisión.

# La matriz de confusión, permite evaluar la cantidad de falsos positivos, es decir, solicitudes 
# validas evaluadas como ataques y la cantidad de falsos negativos, es decir, ataques evaluados 
# como solicitudes válidas.
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
f = open("confusion_matrix", 'w')
f.write(str(matrix))
f.close()

# El classification_report da información estadística sobre las evaluaciones, indicando el recall 
# (la cantidad de verdaderos positivos sobre los positivos detectados), la precisión (los 
# verdaderos positivos sobre la suma de los verdaderos positivos y los falsos positivos) y el F1 
# score (una medida estadística que representa tanto la precisión como el recall en una sola 
# métrica).
report = classification_report(Y_test, Y_pred)
print(report)
f = open("classification_report", 'w')
f.write(str(report))
f.close()

# El accuracy_score es el porcentaje de predicciones correctas sobre el total.
score = accuracy_score(Y_test, Y_pred)
print(score)
f = open("accuracy_score", 'w')
f.write(str(score))
f.close()

# Por último, el script genera el modelo en formato PMML y lo guarda en el archivo "Modelo.pmml".
pmml_destination_path = "Modelo.pmml"
sklearn2pmml(pipeline, pmml_destination_path)