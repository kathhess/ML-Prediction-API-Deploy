import pytest
from ie_bike_model.model import train_and_persist, predict
import os

"""
Adding basic tests to the library: 
- checking that `train_and_persist` 
produces a file in the desired directory
- `predict` returns a positive number for some input data
"""


def test_model_save():
    try:  # if file exists delete
        os.remove("model.joblib")
    except FileNotFoundError:
        pass

    train_and_persist()  # run train_and_persist
    assert os.path.isfile("model.joblib") is True  # confirm file is created


# predict returns positive integer; if file not found create file
# note that delete forces file to not be found
@pytest.mark.parametrize("delete", [(True), (False)])
def test_predict_pos(delete):
    if delete is True:
        try:  # if file exists delete
            os.remove("model.joblib")
        except FileNotFoundError:
            pass

    prediction = predict(
        dteday="2012-11-01",
        hr=10,
        weathersit="Clear, Few clouds, Partly cloudy, Partly cloudy",
        temp=0.3,
        atemp=0.31,
        hum=0.8,
        windspeed=0.0,
    )
    assert prediction > 0
