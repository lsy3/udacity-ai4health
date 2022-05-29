import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df_ret = df.join(ndc_df.set_index('NDC_Code'), on='ndc_code')
    # df_ret['generic_drug_name'] = df_ret['Proprietary Name']
    df_ret['generic_drug_name'] = df_ret['Non-proprietary Name']
    # df_ret['generic_drug_name'] = df_ret['Proprietary Name'] + ' ' + df_ret['Non-proprietary Name']

    return df_ret

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    # simplistic implementation below.
    # however, this takes the first row of groupby patient_nbr, not all the first patient_nbr and encounter_id
    # for example, for the first 5 rows below, the script below shall only take the first row.
    # where the instruction wants us to give the first three rows with the 'first patient_nbr and encounter_id pair'
    #
    # patient_nbr   encounter_id    ndc_code
    # 135           24437208        42291-559
    # 135           24437208        0093-9364
    # 135           24437208        68071-1701
    # 135           26264286        0087-6070
    # 135           26264286        0093-9477
    #
    # first_encounter_df = df.sort_values(['patient_nbr', 'encounter_id']).groupby('patient_nbr').first().reset_index()

    # correct but more complicated script
    first_encounter_df = df.sort_values(['patient_nbr', 'encounter_id'])
    
    index = pd.MultiIndex.from_frame(first_encounter_df[['patient_nbr', 'encounter_id']] \
        .groupby('patient_nbr').first().reset_index()) # .to_records()
    # print(index)
    
    first_encounter_df = first_encounter_df.set_index(['patient_nbr', 'encounter_id'])

    return first_encounter_df.loc[index].reset_index()


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr', valid_percentage=0.2, test_percentage=0.2):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df = df.iloc[np.random.permutation(len(df))]
    
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    
    training_idx = round(total_values * (1.0 - valid_percentage - test_percentage ))
    valid_idx = round(total_values * (1 - test_percentage ))
    
    train = df[df[patient_key].isin(unique_values[:training_idx])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[training_idx:valid_idx])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_values[valid_idx:])].reset_index(drop=True)
    
    return train, validation, test

#Question 7
def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        feature_vocab = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        
        tf_categorical_feature_column = tf.feature_column.indicator_column(feature_vocab)
        # tf_categorical_feature_column = tf.feature_column.embedding_column(feature_vocab, dimension=10)

        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    import functools
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, 
        default_value = default_value, normalizer_fn=normalizer, dtype=tf.float64)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col, threshold=5):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(lambda x: 1 if x >=threshold else 0).to_numpy()
    return student_binary_prediction
