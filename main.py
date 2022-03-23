import json

from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

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
        parser.add_argument('cap-shape')
        parser.add_argument('cap-color')
        parser.add_argument('stalk-shape')
        parser.add_argument('gill-color')
        parser.add_argument('odor')
        parser.add_argument('bruises')
        parser.add_argument('population')

        args = parser.parse_args()  # creates dict
        user_dataframe = pd.DataFrame.from_dict([args])
        #
        # out_file = open("args.json", "w")
        # json.dump(user_dataframe.to_json(), out_file, indent=6)
        # out_file.close()

        # np.array(list(encoded_list.items()), dtype=str)
        # convert dict to dataframe and encode text as numerals
        # encoded_list = encoder.encode(user_dataframe)
        #
        # X_new = np.asarray(encoded_list)
        #
        # out = {'Prediction': MUSHROOM_MODEL.predict([X_new])}
        #
        return args, 200


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True, port='5000')
