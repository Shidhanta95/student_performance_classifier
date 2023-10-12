import pandas as pd
from data_ingestion import datasetIngestion

def dataAnalysis():
    df = datasetIngestion()
    # Printing general information of data 
    print(df.shape)
    print(df.head())
    print(df.tail())
    print(df.describe())
    print(df.info())
    print(df.columns)
    
    # distinguishing features between numerical and categorical values
    cat_types = df.select_dtypes("object").columns
    num_types = df.select_dtypes("number").columns
    print("Categorical: ",cat_types)
    print("Numerical: ",num_types)
    
    
    # displaying total null values in column 
    print(df.isnull().sum(axis = 0))

    #Unique values
    for col in df.columns:
        print(col, df[col].nunique())

    return df

dataAnalysis()