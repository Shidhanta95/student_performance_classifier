import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# import missingno as msno
from datapreprocessing import dataPrepocessing

def dataVisualisation():
    df = dataPrepocessing()

    # distinguishing features between numerical and categorical values
    cat_types = df.select_dtypes("object").columns
    num_types = df.select_dtypes("number").columns
    df_num = df[num_types]

    #correlation heatmap
    corr = df_num.corr()
    plt.figure(figsize = (12,12))
    mp = sns.heatmap(corr, linewidth = 1 ,  annot=True, cmap="coolwarm", fmt=".2f")
    plt.show()

    #missing number visualisation
    # msno.bar(df)
    # plt.show()

    #boxplot
    for col in num_types:
        plt.figure(figsize=(5, 5)) 
        sns.boxplot(data=df, x=col)
        plt.xlabel(col)
    plt.show()

    #data balance
    x = df['grade'].value_counts()
    # df['temp'].plot(kind='bar', figsize=(10, 6))
    p = x.plot(kind = 'bar')
    p.set_xlabel("Student Class Grade Distribution")
    p.set_ylabel("Count")
    p.set_title("Student Grade")
    plt.show()

    return df


dataVisualisation()