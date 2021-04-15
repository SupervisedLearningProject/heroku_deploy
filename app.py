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
    try:
        json_ = request.get_json()
        print(json_)
        query = pd.json_normalize(json_)
        model_columns = ['X', 'Y', 'Index_', 'event_unique_id', 'Primary_Offence', 'Occurrence_Date', 'Occurrence_Year', 'Occurrence_Month', 'Occurrence_Day', 'Occurrence_Time', 'Division', 'City',
                         'Location_Type', 'Premise_Type', 'Bike_Make', 'Bike_Model', 'Bike_Type', 'Bike_Speed', 'Bike_Colour', 'Cost_of_Bike', 'Status', 'Hood_ID', 'Neighbourhood', 'Lat', 'Long', 'ObjectId']
        # query = query.reindex(columns=model_columns, fill_value=0)
        print(query)

        col_to_drop = ['X',
                       'Y',
                       'Index_',
                       'event_unique_id',
                       'City',
                       'Location_Type',
                       'Neighbourhood',
                       'Lat',
                       'Long',
                       'ObjectId',
                       'Bike_Model']

        query.drop(col_to_drop, axis=1, inplace=True)

        query['Occurrence_Date'] = pd.to_datetime(
            query['Occurrence_Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        query['Occurrence_Time'] = pd.to_datetime(
            query['Occurrence_Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        query['hour'] = query['Occurrence_Time'].dt.hour
        query['minute'] = query['Occurrence_Time'].dt.minute

        query['day_of_week'] = query['Occurrence_Date'].dt.day_name()

        query.drop(columns=['Occurrence_Date',
                            'Occurrence_Time'], inplace=True)

        # Pickled the unique values of the Bike_Make, Bike_Type, Bike_Colour
        tmp_list_bike_make = query['Bike_Make'].unique()
        tmp_list_bike_colour = query['Bike_Colour'].unique()
        tmp_list_bike_type = query['Bike_Type'].unique()
        
        list_of_bike_color = joblib.load('list_of_bike_color.pkl')
        list_of_bike_make = joblib.load('list_of_bike_make.pkl')
        list_of_bike_type = joblib.load('list_of_bike_type.pkl')

        bike_make_items = [item in tmp_list_bike_make for item in list_of_bike_make]
        if sum(bike_make_items)!=len(tmp_list_bike_make):
            return jsonify({"error": "Please enter the bike make with in the given list."})
        
        bike_type_items = [item in tmp_list_bike_type for item in list_of_bike_type]
        if sum(bike_type_items)!=len(tmp_list_bike_type):
            return jsonify({"error": "Please enter the bike type with in the given list."})
        
        bike_color_items = [item in tmp_list_bike_colour for item in list_of_bike_color]
        if sum(bike_color_items)!=len(tmp_list_bike_colour):
            return jsonify({"error": "Please enter the bike colour with in the given list"})

        res = {}

        if lr:
            try:
                prediction = list(lr.predict(query))
                print({'prediction': str(prediction)})
                res['lr_prediction'] = prediction
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load lr model."})

        if svm:
            try:
                prediction = list(svm.predict(query))
                print({'prediction': str(prediction)})
                res['svm_prediction'] = prediction
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load svm model."})

        if knn:
            try:
                prediction = list(knn.predict(query))
                print({'prediction': str(prediction)})
                res['knn_prediction'] = prediction
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load knn model."})

        if decision_tree:
            try:
                prediction = list(decision_tree.predict(query))
                print({'prediction': str(prediction)})
                res['decision_tree_prediction'] = decision_tree
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load decision_tree model."})

        if random_forest:
            try:
                prediction = list(random_forest.predict(query))
                print({'prediction': str(prediction)})
                res['random_forest_prediction'] = prediction
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load random_forest model."})

        if rendomized_search_lr:
            try:
                prediction = list(rendomized_search_lr.predict(query))
                print({'prediction': str(prediction)})
                res['rendomized_search_lr_prediction'] = prediction
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load rendomized_search_lr model."})

        if rendomized_search_svm:
            try:
                prediction = list(rendomized_search_svm.predict(query))
                print({'prediction': str(prediction)})
                res['rendomized_search_svm_prediction'] = prediction
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load rendomized_search_svm model."})

        if rendomized_search_decision_tree:
            try:
                prediction = list(
                    rendomized_search_decision_tree.predict(query))
                print({'prediction': str(prediction)})
                res['rendomized_search_decision_tree_prediction'] = prediction
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load rendomized_search_decision_tree model."})

        if rendomized_search_knn:
            try:
                prediction = list(rendomized_search_knn.predict(query))
                print({'prediction': str(prediction)})
                res['rendomized_search_knn'] = prediction
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load rendomized_search_knn model."})

        if rendomized_search_random_forest:
            try:
                prediction = list(
                    rendomized_search_random_forest.predict(query))
                print({'prediction': str(prediction)})
                res['rendomized_search_random_forest'] = prediction
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load rendomized_search_random_forest model."})

        # Return the response in json format
        return jsonify(res)
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/prime_model/predict/', methods=['POST', 'GET'])
def prime_model():
    try:
        json_ = request.get_json()
        print(json_)
        query = pd.json_normalize(json_)
        model_columns = ['X', 'Y', 'Index_', 'event_unique_id', 'Primary_Offence', 'Occurrence_Date', 'Occurrence_Year', 'Occurrence_Month', 'Occurrence_Day', 'Occurrence_Time', 'Division', 'City',
                         'Location_Type', 'Premise_Type', 'Bike_Make', 'Bike_Model', 'Bike_Type', 'Bike_Speed', 'Bike_Colour', 'Cost_of_Bike', 'Status', 'Hood_ID', 'Neighbourhood', 'Lat', 'Long', 'ObjectId']
        # query = query.reindex(columns=model_columns, fill_value=0)
        print(query)

        col_to_drop = ['X',
                       'Y',
                       'Index_',
                       'event_unique_id',
                       'City',
                       'Location_Type',
                       'Neighbourhood',
                       'Lat',
                       'Long',
                       'ObjectId',
                       'Bike_Model']

        query.drop(col_to_drop, axis=1, inplace=True)

        query['Occurrence_Date'] = pd.to_datetime(
            query['Occurrence_Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        query['Occurrence_Time'] = pd.to_datetime(
            query['Occurrence_Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        query['hour'] = query['Occurrence_Time'].dt.hour
        query['minute'] = query['Occurrence_Time'].dt.minute

        query['day_of_week'] = query['Occurrence_Date'].dt.day_name()

        query.drop(columns=['Occurrence_Date',
                            'Occurrence_Time'], inplace=True)

        # Pickled the unique values of the Bike_Make, Bike_Type, Bike_Colour
        tmp_list_bike_make = query['Bike_Make'].unique()
        tmp_list_bike_colour = query['Bike_Colour'].unique()
        tmp_list_bike_type = query['Bike_Type'].unique()
        
        list_of_bike_color = joblib.load('list_of_bike_color.pkl')
        list_of_bike_make = joblib.load('list_of_bike_make.pkl')
        list_of_bike_type = joblib.load('list_of_bike_type.pkl')

        bike_make_items = [item in tmp_list_bike_make for item in list_of_bike_make]
        if sum(bike_make_items)!=len(tmp_list_bike_make):
            return jsonify({"error": "Please enter the bike make with in the given list."})
        
        bike_type_items = [item in tmp_list_bike_type for item in list_of_bike_type]
        if sum(bike_type_items)!=len(tmp_list_bike_type):
            return jsonify({"error": "Please enter the bike type with in the given list."})
        
        bike_color_items = [item in tmp_list_bike_colour for item in list_of_bike_color]
        if sum(bike_color_items)!=len(tmp_list_bike_colour):
            return jsonify({"error": "Please enter the bike colour with in the given list"})
        print(query.columns)
        res = {}
        lr = joblib.load('group1_lr_2021.pkl')
        if lr:
            try:
                prediction = list(lr.predict(query))
                print({'prediction': str(prediction)})
                res['lr_prediction'] = int(prediction)
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load lr model."})
        return jsonify(res)
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route("/test", methods=['GET'])
def test_temp_data():
    res = {}
    test_data = joblib.load('test_data.pkl')    
    if test_data is not None:
        query = test_data
        lr = joblib.load('group1_lr_2021.pkl')
        if lr:
            try:
                prediction = list(lr.predict(query))
                print({'prediction': str(prediction)})
                res['lr_prediction'] = int((prediction))
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({"error": "Error in load lr model."})

#         if svm:
#             try:
#                 prediction = list(svm.predict(query))
#                 print({'prediction': str(prediction)})
#                 res['svm_prediction'] = prediction
#             except:
#                 return jsonify({'trace': traceback.format_exc()})
#         else:
#             return jsonify({"error": "Error in load svm model."})

#         if knn:
#             try:
#                 prediction = list(knn.predict(query))
#                 print({'prediction': str(prediction)})
#                 res['knn_prediction'] = prediction
#             except:
#                 return jsonify({'trace': traceback.format_exc()})
#         else:
#             return jsonify({"error": "Error in load knn model."})

#         if decision_tree:
#             try:
#                 prediction = list(decision_tree.predict(query))
#                 print({'prediction': str(prediction)})
#                 res['decision_tree_prediction'] = decision_tree
#             except:
#                 return jsonify({'trace': traceback.format_exc()})
#         else:
#             return jsonify({"error": "Error in load decision_tree model."})

#         if random_forest:
#             try:
#                 prediction = list(random_forest.predict(query))
#                 print({'prediction': str(prediction)})
#                 res['random_forest_prediction'] = prediction
#             except:
#                 return jsonify({'trace': traceback.format_exc()})
#         else:
#             return jsonify({"error": "Error in load random_forest model."})

#         if rendomized_search_lr:
#             try:
#                 prediction = list(rendomized_search_lr.predict(query))
#                 print({'prediction': str(prediction)})
#                 res['rendomized_search_lr_prediction'] = prediction
#             except:
#                 return jsonify({'trace': traceback.format_exc()})
#         else:
#             return jsonify({"error": "Error in load rendomized_search_lr model."})

#         if rendomized_search_svm:
#             try:
#                 prediction = list(rendomized_search_svm.predict(query))
#                 print({'prediction': str(prediction)})
#                 res['rendomized_search_svm_prediction'] = prediction
#             except:
#                 return jsonify({'trace': traceback.format_exc()})
#         else:
#             return jsonify({"error": "Error in load rendomized_search_svm model."})

#         if rendomized_search_decision_tree:
#             try:
#                 prediction = list(
#                     rendomized_search_decision_tree.predict(query))
#                 print({'prediction': str(prediction)})
#                 res['rendomized_search_decision_tree_prediction'] = prediction
#             except:
#                 return jsonify({'trace': traceback.format_exc()})
#         else:
#             return jsonify({"error": "Error in load rendomized_search_decision_tree model."})

#         if rendomized_search_knn:
#             try:
#                 prediction = list(rendomized_search_knn.predict(query))
#                 print({'prediction': str(prediction)})
#                 res['rendomized_search_knn'] = prediction
#             except:
#                 return jsonify({'trace': traceback.format_exc()})
#         else:
#             return jsonify({"error": "Error in load rendomized_search_knn model."})

#         if rendomized_search_random_forest:
#             try:
#                 prediction = list(
#                     rendomized_search_random_forest.predict(query))
#                 print({'prediction': str(prediction)})
#                 res['rendomized_search_random_forest'] = prediction
#             except:
#                 return jsonify({'trace': traceback.format_exc()})
#         else:
#             return jsonify({"error": "Error in load rendomized_search_random_forest model."})
    else:
        return jsonify({"error": "Error in load test data model."})
    # Return the response in json format
    return jsonify(res)

# A welcome message to test our server


@app.route('/')
def index():
    return "<h1>Welcome to ML Project server !!</h1>"


if __name__ == '__main__':
    lr = joblib.load('group1_lr_2021.pkl')  # Load "model.pkl"
    svm = joblib.load('group1_svm_2021.pkl')  # Load "model.pkl"
    decision_tree = joblib.load(
        'group1_decision_tree_2021.pkl')  # Load "model.pkl"
    knn = joblib.load('group1_knn_2021.pkl')  # Load "model.pkl"
    random_forest = joblib.load(
        'group1_random_forest_2021.pkl')  # Load "model.pkl"

    rendomized_search_lr = joblib.load('group1_randomize_lr_2021.pkl')
    rendomized_search_svm = joblib.load('group1_randomize_svm_2021.pkl')
    rendomized_search_decision_tree = joblib.load(
        'group1_randomize_decision_tree_2021.pkl')
    rendomized_search_knn = joblib.load('group1_randomize_knn_2021.pkl')
    rendomized_search_random_forest = joblib.load(
        'group1_randomize_random_forest_2021.pkl')

    list_of_bike_color = joblib.load('list_of_bike_color.pkl')
    list_of_bike_make = joblib.load('list_of_bike_make.pkl')
    list_of_bike_type = joblib.load('list_of_bike_type.pkl')

    test_data = joblib.load('test_data.pkl')
    if not test_data:
        print('Test Data is not loaded.')

    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000, debug=True)
