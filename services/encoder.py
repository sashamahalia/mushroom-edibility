# convert the data into numeric format
from sklearn import preprocessing


def encode(data):
    le = preprocessing.LabelEncoder()

    cap_shape = le.fit_transform(list(data["cap-shape"]))
    cap_color = le.fit_transform(list(data["cap-color"]))
    odor = le.fit_transform(list(data["odor"]))
    bruises = le.fit_transform(list(data["bruises"]))
    population = le.fit_transform(list(data["population"]))
    stalk_shape = le.fit_transform(list(data["stalk-shape"]))
    gill_color = le.fit_transform(list(data["gill-color"]))

    encoded_list = [cap_shape, cap_color, odor, bruises, population, stalk_shape, gill_color]

    if "edibility" in data:
        edibility = le.fit_transform(list(data["edibility"]))
        encoded_list.append(edibility)
    return encoded_list
