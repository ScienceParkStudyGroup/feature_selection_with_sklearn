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

    # get a Numpy array of the feature names 
    col_names = list(df.columns)
    feature_names = col_names[2:]

    # get a Pandas dataframe of the feature values (e.g. gene expression values)
    features = df.drop(columns=["sample_id","sample_class"], axis=1)

    # get a Numpy array of the sample classes
    labels = df.loc[:,"sample_class"]
    
    return features, labels, feature_names

def random_forest_with_stratified_kfold_cv(kfolds, features, labels, rand_state=123):
    """
    Fits k-fold Random Forest classifiers and returns model QC metrics and feature importances. 

    Parameters
    ----------
    kfolds : int, default=5
        number of folds to create. 
    features:  a Pandas dataframe as created by the import_data_and_separate_features_from_labels of shape (n_samples, n_features)
        the input features from which the trees will be created.
    labels: list of label class names 
       the class labels of the samples of length n_samples
    rand_state: int, default=123 
        RandomState instance used to perform both sample bootstrapping and feature choice at each split. 
    
    Returns
    --------
    mean_test_score: float
    mean_feature_importances: a Pandas dataframe

    training and test sets. It uses these sets to optimize the hyperparameters and return the best model's parameters. 
    A new model is built with the best parameters on the entire training set. After testing, training score, test score, 
    feature importances are returned. 
    
    """
    model = RandomForestClassifier(random_state=rand_state)
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=rand_state) 
    
    scores_test= [] 
    
    feature_importances = []
    
    for train_index, test_index in skf.split(features, labels):
        
        # X are the features and y are the labels (trichome densities)
        X_train, X_test = features.iloc[train_index,:], features.iloc[test_index,:]
        y_train, y_test = labels[train_index], labels[test_index]
       
        #training the model with fit
        model.fit(X_train, y_train)
        
        #saving the gini importance (feature importance)
        feature_importances.append(model.feature_importances_)
        
        #saving the test score of each cv (within each n_estimator value) to a list of scores 
        ## specific to regression 
        model_score = model.score(X_test, y_test)
        scores_test.append(model_score)
        
    #take the mean per n_estimator value 
    mean_test_score = np.average(scores_test)
    sd_test_score = np.std(scores_test)
    mean_feature_importances = np.average(feature_importances, axis=0) 
    
    ## get the averages and std of the means scores and the feature importances 
    
    return mean_test_score, sd_test_score, mean_feature_importances
