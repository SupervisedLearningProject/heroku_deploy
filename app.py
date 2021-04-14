import traceback
import pandas as pd
import joblib
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            model_columns = ['X', 'Y', 'Index_', 'event_unique_id', 'Primary_Offence', 'Occurrence_Date', 'Occurrence_Year', 'Occurrence_Month', 'Occurrence_Day', 'Occurrence_Time', 'Division', 'City', 'Location_Type', 'Premise_Type', 'Bike_Make', 'Bike_Model', 'Bike_Type', 'Bike_Speed', 'Bike_Colour', 'Cost_of_Bike', 'Status', 'Hood_ID', 'Neighbourhood', 'Lat', 'Long', 'ObjectId']
            # query = query.reindex(columns=model_columns, fill_value=0)
            # print(query)



            response = {}

            response["prediction"] = ''

            # Return the response in json format
            return jsonify(response)
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

@app.route("/test",methods=['GET', 'POST'])
def test_temp_data():
    json_ = request.get_json()
    print(json_)
    query = pd.json_normalize(json_)
    print(query)
    
    res = {}
    print(query['id'])
    response = jsonify(res)

    return response

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to ML Project server !!</h1>"

if __name__ == '__main__':
    lr = joblib.load('group1_lr_2021.pkl') # Load "model.pkl"
    print ('Model loaded')
    
    test_data = joblib.load('test_data.pkl')
    if not test_data:
        print('Test Data is not loaded.')

    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000, debug=True)
