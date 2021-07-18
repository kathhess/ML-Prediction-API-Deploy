# ML Model Packaging & Deployment

The following repository is a small project undertaken to re-work and deploy a .ipynb machine learning notebook to production for real-time prediction. 

**To achieve this we:**
- Transform our .ipynb into a Python package using Flit
- Work our cleaned package into an application using Flask
- Deploy our application to Heroku to be used as an API endpoint for ML predictions

## Functionality

- All code is PEP8 compliant
- After training, model weights will be persisted
- If a model is not previously trained, any call of predict will train and save weights
- `$ pytest` can be used to run development tests on model.py during development and changes
- `$ python app.py NNNN` launches and runs our Flask application on port NNNN (host 0.0.0.0); alternatively, port can be specified on PORT environment variable (otherwise defaults to 5001)
- Error handling implemented to return custom status codes for 400, 404, 500 and incorrect model variables passed on URL parameters
- Heroku application is integrated with Github so that updates deploy to live endpoint: 

## .ipynb
The original .ipynb & dataset used to train this model can be found at /reference

## Package Usage

To install from /library:

```
$ pip install .
```

Basic usage:

```python
>>> from ie_bike_model.model import train_and_persist, predict
>>> train_and_persist()  # Trains the model and saves it to `model.joblib`
>>> predict(
...     dteday="2012-11-01",
...     hr=10,
...     weathersit="Clear, Few clouds, Partly cloudy, Partly cloudy"
...     temp=0.3,
...     atemp=0.31,
...     hum=0.8,
...     windspeed=0.0,
... )
105
```

## Local Application

```
$ pip install -r requirements.txt
$ python app.py 5000
```

**Training:**
Open http://0.0.0.0:5000/train in your browser.

**Predition:**
Sample request: http://0.0.0.0:5000/predict?date=2012-11-01&hour=10&weather_situation=clear&temperature=0.3&feeling_temperature=0.31&humidity=0.8&windspeed=0.0

## Deployed Rest API

