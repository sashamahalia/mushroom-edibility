import json

from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from constants import ATTRIBUTES
from sklearn import preprocessing

from services import encoder

APP = Flask(__name__)
cors = CORS(APP, resources={r"/*": {"origins": "*"}})
API = Api(APP)

pickle_in = open("mushrooms.pickle", "rb")
MUSHROOM_MODEL = pickle.load(pickle_in)

class Predict(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        for attribute in ATTRIBUTES:
            parser.add_argument(attribute)

        args = parser.parse_args()  # creates dict
        user_dataframe = pd.DataFrame.from_dict([args])

        # convert dict to dataframe and encode text as numerals
        encoded_list = encoder.encode(user_dataframe)

        # X_new = np.fromiter(encoded_list, dtype=float)
        # print(X_new)
        # print(encoded_list[0])
        # out = {'Prediction': MUSHROOM_MODEL.predict(X_new)}
        # print(out)
        return user_dataframe.to_json(), 200


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True, port='5000')
