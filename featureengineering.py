import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from datapreprocessing import dataPrepocessing
import numpy as np


def dataEngineering():
    
    data = dataPrepocessing()
    percentile25 = data['absences'].quantile(0.25)
    percentile75 = data['absences'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    data['absences'] = np.where(
    data['absences'] > upper_limit,
        upper_limit,
        np.where(
            data['absences'] < lower_limit,
            lower_limit,
            data['absences']
        ))
    data['absences'] = data['absences'].astype(int) 
    
    lab = LabelEncoder()
    categorical_features = [feature for feature in data.columns if ((data[feature].dtypes=='O') & (feature not in ['y']))]
    
    for column in categorical_features:
        data[column] = lab.fit_transform(data[column])
    data.to_csv("data_mod.csv", index = False)
    #print(X.columns())
    return data


dataEngineering()
