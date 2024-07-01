import joblib
import pandas as pd
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

class Model:
    MODEL_PATH = "data/model.joblib"
    ENCODER_PATH = "data/encoder.joblib"
    AREA_ENCODER_PATH = "data/area_encoder.joblib"
    ITEM_ENCODER_PATH = "data/item_encoder.joblib"

    def __init__(self):
        self.loaded = False
        self.model = LinearRegression()
        self.encoder = OrdinalEncoder()
        self.area_encoder = OrdinalEncoder()
        self.item_encoder = OrdinalEncoder()

    def predict(self, area, item, rainfall, pesticides, temperature):
        if self.loaded:
            x = np.array([area, item, rainfall, pesticides, temperature])
            x = self.encoder.inverse_transform(x.reshape(1, -1))
            return self.model.predict(x)[0]
        else:
            raise Exception("Model not loaded or trained")
            

    def train(self, dataset_path):
        df = pd.read_csv(dataset_path)

        X = df[['Area', 'Item', 'average_rain_fall_mm_per_year'	,'pesticides_tonnes','avg_temp']] 
        Y = df['hg/ha_yield']

        self.encoder.fit(X)

        X = self.encoder.transform(X)

        # area = X["Area"].to_list()
        # self.area_encoder.fit(area)
        # area = self.area_encoder.transform(area)
        # X["Area"] = area

        # item = df["Item"].to_list()
        # self.item_encoder.fit(item)
        # item = self.item_encoder.transform(item)
        # X['Item'] = item

        X = X
        Y = Y.to_numpy()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=69)

        self.model.fit(X_train, Y_train)

        Y_pred = self.model.predict(X_test)

        print('Coefficients: ', self.model.coef_)
        print('MSE: ', mean_squared_error(Y_test, Y_pred))

        self.loaded = True
        

    def save(self):
        os.makedirs("data", exist_ok=True)

        joblib.dump(self.model, Model.MODEL_PATH)
        joblib.dump(self.encoder, Model.ENCODER_PATH)
        joblib.dump(self.area_encoder, Model.AREA_ENCODER_PATH)
        joblib.dump(self.item_encoder, Model.ITEM_ENCODER_PATH)

    def load(self):
        self.model = joblib.load(Model.MODEL_PATH)
        self.encoder = joblib.load(Model.ENCODER_PATH)
        self.area_encoder = joblib.load(Model.AREA_ENCODER_PATH)
        self.item_encoder = joblib.load(Model.ITEM_ENCODER_PATH)

        self.loaded = True
    

