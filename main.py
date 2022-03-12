from flask import Flask
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np

APP = Flask(__name__)
API = Api(APP)

pickle_in = open("mushrooms.pickle", "rb")
MUSHROOM_MODEL = pickle.load(pickle_in)


class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('cap_shape')
        parser.add_argument('cap_color')
        parser.add_argument('odor')
        parser.add_argument('bruises')
        parser.add_argument('population')
        parser.add_argument('stalk_shape')
        parser.add_argument('gill_color')

        args = parser.parse_args()  # creates dict

        X_new = np.fromiter(args.values(), dtype=float)  # convert input to array

        out = {'Prediction': MUSHROOM_MODEL.predict([X_new])[0]}

        return out, 200
