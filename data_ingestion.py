import pandas as pd

def datasetIngestion():
    df = pd.read_csv('data.csv')
    return df

datasetIngestion()