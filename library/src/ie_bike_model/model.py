import os
import pandas as pd
import numpy as np
from joblib import dump, load

# MODEL PACKAGES
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FILE_PATH, "hour.csv")

df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
X = df.drop(columns=["instant", "cnt", "casual", "registered"])
y = df["cnt"]


def train_and_persist():
    """
    Input: Data
    Output: Trained model
    """

    # PREPROCESS
    preprocessing = preprocess()

    # TRAIN MODEL
    reg = Pipeline(
        [("preprocessing", preprocessing), ("model", RandomForestRegressor())]
    )

    X_train, y_train = X.loc[X["dteday"] < "2012-10"], y.loc[X["dteday"] < "2012-10"]
    X_test, y_test = X.loc[X["dteday"] >= "2012-10"], y.loc[X["dteday"] >= "2012-10"]

    # RETURN FIT
    reg.fit(X_train, y_train)

    # EVALUATE
    y_pred = reg.predict(X_test)
    rmse = mean_squared_error(y_pred=y_pred, y_true=y_test, squared=False)

    # SAVE
    dump(reg, "model.joblib")

    return rmse


def predict(dteday, hr, weathersit, temp, atemp, hum, windspeed):
    """
    Input:
        predict(
        dteday="2012-11-01",
        hr=10,
        weathersit="Clear, Few clouds, Partly cloudy, Partly cloudy"
        temp=0.3,
        atemp=0.31,
        hum=0.8,
        windspeed=0.0,)
    Output: Single predicted value

    If model file exists, load and use; else train with default data before predicting.
    The `predict` function should return a single integer.

    """

    # RELOADMODEL
    # if model file doesn't exist, create before loading
    try:
        reg = load("model.joblib")
    except FileNotFoundError:
        train_and_persist()
        reg = load("model.joblib")

    # FORMAT DATA
    inputs = [pd.to_datetime(dteday), hr, weathersit, temp, atemp, hum, windspeed]
    columns = ["dteday", "hr", "weathersit", "temp", "atemp", "hum", "windspeed"]
    new_df = pd.DataFrame([inputs], columns=columns)

    # PREDICTION
    prediction = reg.predict(new_df)

    return round(prediction[0])  # predict function should return single integer


def preprocess():
    ffiller = FunctionTransformer(ffill_missing)

    weather_enc = make_pipeline(
        ffiller,
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=X["weathersit"].nunique()
        ),
    )

    ct = make_column_transformer(
        (ffiller, make_column_selector(dtype_include=np.number)),
        (weather_enc, ["weathersit"]),
    )

    preprocessing = FeatureUnion(
        [
            ("is_weekend", FunctionTransformer(is_weekend)),
            ("year", FunctionTransformer(year)),
            ("column_transform", ct),
        ]
    )

    return preprocessing


def ffill_missing(ser):
    return ser.fillna(method="ffill")


def is_weekend(data):
    return data["dteday"].dt.day_name().isin(["Saturday", "Sunday"]).to_frame()


def year(data):
    # Our reference year is 2011, the beginning of the training dataset
    return (data["dteday"].dt.year - 2011).to_frame()


if __name__ == "__main__":
    print(
        predict(
            dteday="2012-11-10",
            hr=10,
            weathersit="Clear, Few clouds, Partly cloudy, Partly cloudy",
            temp=0.3,
            atemp=0.31,
            hum=0.8,
            windspeed=0.0,
        )
    )
