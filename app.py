"""
Simple post server for model implementation
"""
from flask import Flask, request, jsonify, make_response
from sklearn.externals import joblib
import pandas as pd
import re

application = Flask(__name__)

@application.route('/', methods = ['POST'])


def index():
    input_feature_dict = {
        'body_style_label': pd.read_csv('body_style_label.csv'),
        'location_label': pd.read_csv('location_label.csv'),
        'ticket_number_label': pd.read_csv('ticket_number_label.csv'),
        'violation_code_label': pd.read_csv('violation_code_label.csv'),
        'color_label': pd.read_csv('color_label.csv'),
        'violation_description_label': pd.read_csv('violation_description_label.csv')
    }
    input_features = ['body_style_label',
        'location_label',
        'ticket_number_label',
        'Latitude',
        'Longitude',
        'violation_code_label',
        'color_label',
        'violation_description_label']
   
    pred_model = joblib.load('random_forest_classifier.pkl')

    if request.method == 'POST':
        data = request.form.to_dict()

        data_dict = eval(list(data.keys())[0])

        df_data = pd.DataFrame(columns=list(data_dict.keys()))
        for icol in df_data.columns:
            df_data[icol] = [data_dict[icol]]
            if icol not in ['Latitude', 'Longitude']:
                icol_name = re.sub(' ', '_', icol.lower().strip()) 
                icol_label = icol_name + '_label'
                icol_label_dict = input_feature_dict[
                    icol_label].set_index(icol_name)[icol_label].to_dict()
                df_data[icol_label] = df_data.apply(
                    lambda row: icol_label_dict[str(row[icol])],
                    axis=1)
        
        df_data = df_data[input_features]
        input_array = df_data.values

        results = pred_model.predict_proba(input_array)[0]
        results = [round(ii, 3) for ii in results]

        df_make = pd.read_csv('make_label.csv')
        makes = sorted(set(df_make.make))

        results_dict = dict(zip(makes, results))

        results_output = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)

        output = {'prediction': results_output}
        
        #print(output)

        return make_response(jsonify(output))

if __name__ == '__main__':
    application.run()

"""
curl -X POST http://127.0.0.1:5000/ -d "{'Body Style': 'PA', 'Location': 'PLATA/RAMPART', 'Ticket number': 1107780811, 'Latitude': 99999.0, 'Longitude': 99999.0, 'Violation code': '8069B', 'Color': 'BK', 'Violation Description':'NO PARKING'}"
"""
