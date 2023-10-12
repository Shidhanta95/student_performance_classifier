import pandas as pd
import numpy as np
from data_ingestion import datasetIngestion
from scipy import stats

def dataPrepocessing():
    data = datasetIngestion()

    data['score'] = ((data["G1"]+data["G2"]+data["G3"])/60)*100
    # create a list of our conditions
    conditions = [
        (data['score'] <= 69),
        (data['score'] >= 70) & (data['score'] <= 89),
        (data['score'] >= 90)
    ]

    # create a list of the values we want to assign for each condition
    values = ['L', 'M', 'H']

    # create a new column and use np.select to assign values to it using our lists as arguments
    data['grade'] = np.select(conditions, values)

    #drop column
    #cols = [df.columns[0]]
    #cols = [df.columns[0],'school','sex','age','address','famsize','Medu','Pstatus','guardian','schoolsup','famsup','paid','activities','nursery','romantic','famrel','freetime','goout','Dalc','Walc','health']
    #df.drop(cols,inplace=True, axis=1)
    
    data.drop(['Unnamed: 0', 'school', 'Mjob', 'Fjob', 'reason', 'guardian', 'nursery', 'romantic', 'famrel', 'Dalc', 'Walc', 'G1', 'G2', 'G3','score'], axis=1, inplace=True)
    print("Duplicate Rows: ",data.duplicated().sum())

    #Exploring the Categorical Features
    categorical_features = [feature for feature in data.columns if ((data[feature].dtypes=='O') & (feature not in ['y']))]
    print(categorical_features)

    for column in categorical_features:
        counts = data[column].value_counts()
        print(counts)

    return data


dataPrepocessing()

