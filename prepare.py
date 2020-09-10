import pandas as pd
import numpy as np
import scipy as sp 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# prep_iris 
def prep_iris(df):
    df = df.drop(columns='species_id')
    df = df.rename(columns={'species_name': 'species'})
    df_dummies = pd.get_dummies(df[['species']], drop_first=[True, True])
    df = pd.concat([df, df_dummies], axis=1)
    return df

# prep_titanic

def train_valid_test(df):
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123, stratify = df.survived)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123, stratify = train_validate.survived)
    return train, validate, test

def impute_age(train, validate, test):
    avg_age = train.age.mean()
    train.age = train.age.fillna(avg_age)
    validate.age = validate.age.fillna(avg_age)
    test.age = test.age.fillna(avg_age)
    return train, validate, test

def prep_titanic(df):
    df.drop(columns = ['embarked', 'class', 'passenger_id', 'deck'], inplace = True)
    # drop missing observations of embark town
    df = df[~df.embark_town.isnull()]
    df_dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first = True)
    df_new = pd.concat([df, df_dummies], axis = 1)
    #split data
    train, validate, test = train_valid_test(df_new)
    #impute age data
    impute_age(train, validate, test)
    return train, validate, test

