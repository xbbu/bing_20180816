"""
Build model to predict probability for top 25 car makes
"""
import pandas as pd
import requests
import time
import math
from scipy.stats import chi2_contingency, chi2
import re
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import pickle as pickle
from sklearn.metrics import accuracy_score

from pull_data import get_csv_from_url

# source data file
data_file = 'https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv'

# load data
print('*** loading data ......')
#df = pd.read_csv(requests.get(data_file).content)
df = get_csv_from_url(data_file)

# removing missing values/rows
print('*** cleaning data .....')
continuous_vars = ['Fine amount', 'Latitude', 'Longitude']

useless_vars = ['Meter Id', 'Marked Time', 'VIN']

categorical_vars = sorted(set(df.columns))

categorical_vars = [
    ii for ii in categorical_vars if ii not in continuous_vars]

for ivar in useless_vars:
    if ivar in categorical_vars:
        del df[ivar]
        categorical_vars.remove(ivar)
    
for ivar in categorical_vars:
    df[ivar] = df[ivar].apply(lambda x: str(x))
    df = df[df[ivar] != "nan"] # read file from local csv 
    df = df[df[ivar] != ''] # read file from web link


# keep data for top 25 makes only
print('*** selecting top 25 makes only ......')
top_makes = list(df.Make.value_counts()[:25].index)
df_top = pd.DataFrame(top_makes, columns=['Make'])
df = df_top.merge(df, on='Make', how='inner')


# add a new variable to mark if the plate is expired when get a ticket
# which reflects part info of the driver
print('*** adding customer transformer ......')
df['date_ticket'] = df['Issue Date'].apply(
    lambda x: x.split('T')[0])
def get_date(x):
    x = x.lower().strip()
    x = str(x).split('.')[0]
    if len(x) != 6:
        return 'none'
    x = x[:4] + '-' + x[4:] + '-01'
    return x

df['date_expire'] = df['Plate Expiry Date'].apply(
    lambda x: get_date(x))

print(len(df))
df = df.query('date_expire != "none"')
print(len(df))

df['is_expired'] = df.apply(
    lambda row: 1 if row['date_ticket'] > row['date_expire'] else 0,
    axis=1)

# use this as example to explore the correlations,
# which is not optimal for categorical variables
print('*** exploring correlations ......')
label_encoder = LabelEncoder()
for ivar in categorical_vars:
    #if ivar != 'Agency': continue
    ivar_name = re.sub(' ', '_', ivar.lower().strip())
    ivar_label = '{}_label'.format(ivar_name)
    #print(ivar_label)
    df[ivar_label] = label_encoder.fit_transform(df[ivar])

print(df.corr(method='kendall')['make_label'])

# feature selection
print('*** feature selection ......')
initial_inputs = [
    'Fine amount', 'Latitude', 'Longitude', 'agency_label',
    'body_style_label', 'color_label', 'is_expired',
    'issue_date_label', 'issue_time_label', 'location_label',
    'plate_expiry_date_label', 'route_label', 'rp_state_plate_label',
    'ticket_number_label', 'violation_code_label', 'violation_description_label']

for ivar in ['Fine amount', 'Latitude', 'Longitude']:
    """ for using local csv file
    df[ivar] = df[ivar].apply(
        lambda x: -1 if math.isnan(x) or type(x) is str else x)
    df = df[df[ivar] > 0.0]
    """
    df[ivar] = df[ivar].apply(lambda x: str(x))
    df = df[df[ivar] != '']
    df = df[df[ivar] != 'nan']
    df[ivar] = df[ivar].apply(lambda x: eval(x))

feature_importance_dict = dict(zip(
    initial_inputs,
    mutual_info_classif(df[initial_inputs], df['make_label'])))

# choose top 8 features as example for training model
features_sorted = sorted(
    feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
features_selected = [ii[0] for ii in features_sorted[:8]]

print(features_selected)

# stratified sampling based on the Make category
print('*** stratified sampling ......')
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=33)
df = df.reset_index(drop=True)
strat_train_set, strat_test_set = pd.DataFrame(), pd.DataFrame()

for train_index, test_index in split.split(df, df['Make']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# model training
print('*** training model ......')
clf = RandomForestClassifier(n_estimators=100, max_depth=2)
clf_model = clf.fit(
    strat_train_set[features_selected], strat_train_set['make_label'])

# save model
print('*** save model to local disk as pickle file ......')
with open('random_forest_classifier.pkl', 'wb') as f:
    pickle.dump(clf_model, f)
    f.close()

# model evaluation
print('*** model evaluation/scoring ......')
y_test_true = strat_test_set['make_label'].values
y_test_pred = clf_model.predict(strat_test_set[features_selected].values)
acc_score = accuracy_score(y_test_true, y_test_pred)
print(acc_score)

# save label transformation to csv file for later model implementation
print('*** saving label transformation file to local csv files ......')
for ivar in categorical_vars:
    ivar_name = re.sub(' ', '_', ivar.lower().strip())
    ivar_label = '{}_label'.format(ivar_name)
    ivar_dict = df.set_index(ivar)[ivar_label].to_dict()
    #print(ivar)
    #print(ivar_dict)

    df_ivar = pd.DataFrame(columns=[ivar_name, ivar_label])
    df_ivar[ivar_name] = list(ivar_dict.keys())
    df_ivar[ivar_label] = df_ivar[ivar_name].apply(
        lambda x: ivar_dict[x])
    df_ivar.to_csv('{}.csv'.format(ivar_label), index=False)

print('*** Done for modeling ***')
