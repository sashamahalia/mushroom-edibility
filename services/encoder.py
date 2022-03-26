# convert the data into numeric format
from sklearn import preprocessing
from constants import ATTRIBUTES


def encode(data):
    print("entering encoder: ", data)
    le = preprocessing.LabelEncoder()

    encoded_list = []

    for key, value in ATTRIBUTES.items():
        if key in data:
            le.fit(value)
            encoded_list.append(le.transform(data[key]))

    print("leaving encoder: ", encoded_list)

    return encoded_list
