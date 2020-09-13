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

###################### Prepare Telco Churn Data ######################

def telco_train_valid_test(df):
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123, stratify = df.churn)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123, stratify = train_validate.churn)
    return train, validate, test

def prep_telco_data(df):
    # Delete columns 'customer_id', contract_type_id, internet_service_type_id, payment_type_id    
    df.drop(columns = ['customer_id','contract_type_id','internet_service_type_id', 'payment_type_id'], inplace = True)
    # Replace partner, dependents, churn, phone_service, paperless billing, with boolean value
    df.partner.replace(['Yes', 'No'], [1,0], inplace = True)
    df.dependents.replace(['Yes', 'No'], [1,0], inplace = True)
    df.churn.replace(['Yes', 'No'], [1,0], inplace = True)
    df.phone_service.replace(['Yes', 'No'], [1,0], inplace = True)
    df.paperless_billing.replace(['Yes', 'No'], [1,0], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #gender = df.gender.str.get_dummies()
    #df = pd.concat([df, gender], axis=1)
    #df.rename(columns = {'Female': 'is_female', 'Male': 'is_male'}, inplace = True)
    #df.drop(columns = ['gender'], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #multiple = df.multiple_lines.str.get_dummies()
    #df = pd.concat([df, multiple], axis=1)
    #df.rename(columns = {'No': 'no_multiple_lines', 'Yes': 'yes_multiple_lines'}, inplace = True)
    #df.drop(columns = ['multiple_lines'], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #multiple = df.online_security.str.get_dummies()
    #df = pd.concat([df, multiple], axis=1)
    #df.rename(columns = {'No': 'no_online_security', 'Yes': 'yes_online_security'}, inplace = True)
    #df.drop(columns = ['online_security'], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #multiple = df.online_backup.str.get_dummies()
    #df = pd.concat([df, multiple], axis=1)
    #df.rename(columns = {'No': 'no_online_backup', 'Yes': 'yes_online_backup'}, inplace = True)
    #df.drop(columns = ['online_backup'], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #multiple = df.device_protection.str.get_dummies()
    #df = pd.concat([df, multiple], axis=1)
    #df.rename(columns = {'No': 'no_device_protection', 'Yes': 'yes_device_protection'}, inplace = True)
    #df.drop(columns = ['device_protection'], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #multiple = df.tech_support.str.get_dummies()
    #df = pd.concat([df, multiple], axis=1)
    #df.rename(columns = {'No': 'no_tech_support', 'Yes': 'yes_tech_support'}, inplace = True)
    #df.drop(columns = ['tech_support'], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #multiple = df.streaming_tv.str.get_dummies()
    #df = pd.concat([df, multiple], axis=1)
    #df.rename(columns = {'No': 'no_streaming_tv', 'Yes': 'yes_streaming_tv'}, inplace = True)
    #df.drop(columns = ['streaming_tv', 'No internet service'], inplace = True)
    #df.drop(columns = ['No internet service'], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #multiple = df.streaming_movies.str.get_dummies()
    #df = pd.concat([df, multiple], axis=1)
    #df.rename(columns = {'No': 'no_streaming_movies', 'Yes': 'yes_streaming_movies'}, inplace = True)
    #df.drop(columns = ['streaming_movies'], inplace = True)
    # Add dummy variables as new columns in dataframe and rename them, delete origional
    multiple = df.contract_type.str.get_dummies()
    df = pd.concat([df, multiple], axis=1)
    df.rename(columns = {'Month-to-month': 'month_to_month_contract', 'One year': 'one_year_contract', 'Two year': 'two_year_contract'}, inplace = True)
    #df.drop(columns = ['contract_type'], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #multiple = df.internet_service_type.str.get_dummies()
    #df = pd.concat([df, multiple], axis=1)
    #df.rename(columns = {'DSL': 'dsl', 'Fiber optic': 'fiber_optic'}, inplace = True)
    #df.drop(columns = ['internet_service_type','None'], inplace = True)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    #multiple = df.payment_type.str.get_dummies()
    #df = pd.concat([df, multiple], axis=1)
    #df.rename(columns = {'Bank transfer (automatic)': 'auto_bank_transfer', 'Credit card (automatic)': 'auto_credit_card', 'Electronic check': 'e_check', 'Mailed check': 'mail_check'}, inplace = True)
    #df.drop(columns = ['payment_type'], inplace = True)
    # Change total_charges to float from object
    df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
    #split data
    train, validate, test = telco_train_valid_test(df)
    return train, validate, test
