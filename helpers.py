#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold



def import_data_and_separate_features_from_labels(csv_file_path):
    """
    Loads the data from a .csv file and returns
    Parameters
    ----------
    csv_file_path : string
    The input is a .csv file path that is used to create a Pandas dataframe
    Samples should be in rows and features in columns. 
    This function separates the data into feature names (e.g. gene names), features (gene expression values), and labels (sample classes). 
    The feature names are in a list, and both features and label are returned as numpy arrays of the values for feature values 
    and sample class. 
    Returns 
    ---------
    1) a N samples x p variables feature Pandas dataframe  
    2) a list with the sample class labels 
    3) a list with the feature names 
    

    Example of input .csv file
    ---------------------------
	sample_id,sample_class,clump,uniformity_cell_size,uniformity_cell_shape,adhesion
	sample_01,benign,5,1,1,1
	sample_02,benign,5,4,4,5
	sample_03,benign,3,1,1,1
	sample_04,benign,6,8,8,1
	sample_05,benign,4,1,1,3
	sample_06,malign,8,10,10
	...[many more rows]
    """
    
    df = pd.read_csv(csv_file_path).dropna(how="all", axis=1)

    # get a list of the feature names 
    col_names = list(df.columns)
    feature_names = col_names[2:]

    # get a Pandas dataframe of the feature values (e.g. gene expression values)
    features = df.drop(columns=["sample_id","sample_class"], axis=1)

    # get a list of the sample classes
    label = df.loc[:,"sample_class"].tolist()
    
    return features, label, feature_names

