from flask import Flask,request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS #which will allow us to access which is not in our newtork basically connect backend and frontend if both are in 2 differnet servers (if)

app = Flask(__name__)
CORS(app)

#load the train models
model=load('./stroke_prediction_model.joblib')

#initalize the flask app
app=Flask(__name__)
CORS(app)

@app.route('/predict',methods=['POST'])  # this is route called predict
def predict():
    try:
        data=request.json #the json data receieved fromt the frontend is stored here
        df=pd.DataFrame([data]) #the data is pacekd into packets or dataframe
        prediction=model.predict(df)[0]
        print(f"Prediction: {prediction}")
        return jsonify({'stroke': int(prediction)}),200


    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the Stroke Prediction API!"

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)