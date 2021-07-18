from flask import Flask, request
from ie_bike_model.model import train_and_persist, predict

app = Flask(__name__)


@app.errorhandler(400)
def bad_request(e):
    return {"status": "error: bad request"}


@app.errorhandler(404)
def page_not_found(e):
    return {"status": "error: page not found"}


@app.errorhandler(500)
def internal_error(e):
    return {"status": "error: internal error"}


@app.route("/")
def home():
    """
    Returns a dictionary with the versions of scikit-learn, ie-bike-model, and Python.
    """
    import sklearn
    import platform
    import ie_bike_model

    version_dict = {}

    version_dict["scikit-learn"] = sklearn.__version__
    version_dict["ie-bike-model"] = ie_bike_model.__version__
    version_dict["Python"] = platform.python_version()
    version_dict["status"] = "ok"

    return version_dict


@app.route("/train")
def do_train_and_persist():
    """
    {"status": "ok"}
    """

    # train_and_persist()
    # return {"status": "ok"}

    response_dict = {"status": "ok"}
    response_dict["rmse"] = train_and_persist()
    return response_dict


@app.route("/predict")
def do_predict():
    """
    INPUT:
    http://0.0.0.0:5000/predict?date=2012-11-01&hour=10&weather_situation=clear&temperature=0.3&feeling_temperature=0.31&humidity=0.8&windspeed=0.0
    OUTPUT:
    `{"result": NN: "elapsed_fime": FF.FFF}`
    """
    import time

    start_time = time.time()

    # extract and map param values from URL string to variables; pass to function
    dteday = request.args["date"]
    hr = request.args["hour"]
    temp = request.args["temperature"]
    atemp = request.args["feeling_temperature"]
    hum = request.args["humidity"]
    windspeed = request.args["windspeed"]
    weather_situation = request.args["weather_situation"]

    try:
        valid_weather_dict = {
            "clear": "Clear, Few clouds, Partly cloudy, Partly cloudy",
            "cloudy": "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
            "light_rain": "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
            "heavy_rain": "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog",
        }

        weathersit = valid_weather_dict[weather_situation]
        prediction = predict(dteday, hr, weathersit, temp, atemp, hum, windspeed)

        # calculate elapsed time
        elapsed = time.time() - start_time

        # store results to dictionary to be returned
        result_dict = {}
        result_dict["status"] = "ok"
        result_dict["result"] = float("{:.0f}".format(prediction))
        result_dict["elapsed_time"] = float("{:.2f}".format(elapsed))

        return result_dict

    except KeyError:
        return {"status": "error: invalid weather_situation"}


if __name__ == "__main__":
    import os
    import sys

    """
    python app.py 5000
    """

    try:  # import port specification from command line
        port = sys.argv[1]

    except IndexError:  # if port isn't specified, use env port
        try:
            port = int(os.environ["PORT"])
        except KeyError:  # if environ port not set use port 5001
            port = 5001

    app.run(port=port, host="0.0.0.0")
