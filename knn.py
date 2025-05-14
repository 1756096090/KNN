import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargando el dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, :-1].values  # Todas las columnas menos la última
y = dataset.iloc[:, -1].values   # La última columna es la variable objetivo

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalado de características
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenamiento del modelo K-NN
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicción de un nuevo resultado (usamos valores arbitrarios escalados)
nuevo = np.array([[5, 116, 74, 0, 0, 25.6, 0.201, 30]])  # valores simulados
resultado = classifier.predict(sc.transform(nuevo))
print(f"¿Tiene diabetes? {'Sí' if resultado[0] == 1 else 'No'}")

# Predicción del conjunto de prueba
y_pred = classifier.predict(X_test)

# Matriz de confusión y precisión
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)
print("Precisión del modelo:")
print(accuracy)

# Visualización (usamos solo 2 primeras características para graficar)
X_vis = X[:, :2]  # Solo las dos primeras columnas para visualización
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.25, random_state=0)
sc_vis = StandardScaler()
X_train_vis = sc_vis.fit_transform(X_train_vis)
X_test_vis = sc_vis.transform(X_test_vis)

classifier_vis = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_vis.fit(X_train_vis, y_train_vis)

# Visualización del conjunto de entrenamiento
X_set, y_set = sc_vis.inverse_transform(X_train_vis), y_train_vis
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.1),
                     np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.1))
plt.contourf(X1, X2, classifier_vis.predict(sc_vis.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('red', 'green')), edgecolor='k')
plt.title('K-NN (Conjunto de Entrenamiento - 2 variables)')
plt.xlabel('Pregnancies')
plt.ylabel('Glucose')
plt.legend(['No Diabetes', 'Diabetes'])
plt.show()

# Visualización del conjunto de prueba
X_set, y_set = sc_vis.inverse_transform(X_test_vis), y_test_vis
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.1),
                     np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.1))
plt.contourf(X1, X2, classifier_vis.predict(sc_vis.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('red', 'green')), edgecolor='k')
plt.title('K-NN (Conjunto de Prueba - 2 variables)')
plt.xlabel('Pregnancies')
plt.ylabel('Glucose')
plt.legend(['No Diabetes', 'Diabetes'])
plt.show()
