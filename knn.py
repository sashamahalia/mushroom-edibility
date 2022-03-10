import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("mushroom-data/agaricus-lepiota.data")

# convert the data into numeric format
le = preprocessing.LabelEncoder()

edibility = le.fit_transform(list(data["edibility"]))
cap_shape = le.fit_transform(list(data["cap-shape"]))
cap_color = le.fit_transform(list(data["cap-color"]))
odor = le.fit_transform(list(data["odor"]))
bruises = le.fit_transform(list(data["bruises"]))
population = le.fit_transform(list(data["population"]))
stalk_shape = le.fit_transform(list(data["stalk-shape"]))
gill_color = le.fit_transform(list(data["gill-color"]))

# setup input and output
X = list(zip(cap_shape, cap_color, odor, bruises, population, stalk_shape, gill_color))
y = list(edibility)

# Adds human readable label for output
labels = {0 : 'edible',
              1 : 'poisonous'}

y = np.vectorize(labels.__getitem__)(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=1)

# fit the model
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

acc = accuracy_score(y_test, predictions)
print(f'Successfully trained model with an accuracy of {acc:.2f}')






