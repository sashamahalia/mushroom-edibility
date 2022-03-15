import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle
from services import encoder

data = pd.read_csv("mushroom-data/agaricus-lepiota.data")

# setup input and output
encoded_list = encoder.encode(data)

X = list(zip(*encoded_list[:-1]))
y = list(encoded_list[-1])

# X = list(zip(cap_shape, cap_color, odor, bruises, population, stalk_shape, gill_color))
# y = list(edibility)

# Adds human readable label for output
labels = {0: 'edible',
          1: 'poisonous'}

y = np.vectorize(labels.__getitem__)(y)


def train(X, y):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    knn = KNeighborsClassifier(n_neighbors=1)

    # fit the model
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    print(f'Successfully trained model with an accuracy of {acc:.2f}')
    return knn


if __name__ == '__main__':
    model = train(X, y)

    # serialize model
    with open("mushrooms.pickle", "wb") as f:
        pickle.dump(model, f)
