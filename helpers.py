#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support



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

def random_forest_with_stratified_kfold_cv(kfolds, features, labels, n_trees = 1000, rand_state=123, score_digits=3, positive_class=None):
    """
    Fits k-fold Random Forest classifiers and returns model QC metrics, feature importances, a confusion matrix for display and precision/recall. 

    Parameters
    ----------
    kfolds : int, default=5
        number of folds to create. 
    features: a Pandas dataframe as created by the import_data_and_separate_features_from_labels of shape (n_samples, n_features)
        the input features from which the trees will be created.
    labels: list of label class names 
       the class labels of the samples of length n_samples
    n_trees: int, default=1000
       The number of trees in the forest.
    rand_state: int, default=123 
        RandomState instance used to perform both sample bootstrapping and feature choice at each split. Set for reproducible results.
    score_digits: int, default=2
        Number of decimals to round the mean accuracy test score of the built Random Forest model
    
    Returns
    --------
    scores: Pandas dataframe with the model train/test accuracy scores
        mean and std accuracy on the given train and test data.
    original_feature_importances: a Pandas dataframe of shape (n_features, 3)
        A Pandas dataframe that contains feature names and their mean and standard deviation of their feature importance computed from the k-fold cv.
    confusion_matrix_disp: a ConfusionMatrixDisplay object ready for plotting
        Call .plot() on the object for display
    pos_label_scores: a Pandas dataframe with precision, recall, fbeta_score and number of occurrences of each label.
   
    """
    model = RandomForestClassifier(n_estimators=n_trees, criterion="gini", random_state=rand_state)
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=rand_state) 
    
    scores_train=[]
    scores_test= []
    feature_importances = []
    true_labels = []      # a list of the true sample labels (one per k-fold)
    predicted_labels = [] # a list of the predicted sample labels (one per k-fold)
    
    for train_index, test_index in skf.split(features, labels):

        # X are the features and y are the labels (sample classes)
        X_train, X_test = features.iloc[train_index,:], features.iloc[test_index,:]
        y_train, y_test = labels[train_index], labels[test_index]
       
        # Training the model with fit
        model.fit(X_train, y_train)
        
        # Saving the gini importance (feature importance)
        feature_importances.append(model.feature_importances_)
        
        # Saving the test score of each cv (within each n_estimator value) to a list of scores
        # Useful to detect overfitting on training dataset 
        model_train_score = model.score(X_train, y_train)
        scores_train.append(model_train_score)
        model_test_score = model.score(X_test, y_test)
        scores_test.append(model_test_score)
        
        ## Computing the confusion matrix
        predicted_labels.append(model.predict(X_test))
        true_labels.append(y_test)

    ### Average/SD of model train/test scores
    mean_train_score = np.around(np.average(scores_train), decimals=score_digits)
    sd_train_score = np.around(np.std(scores_train), decimals=score_digits) 
    mean_test_score = np.around(np.average(scores_test), decimals=score_digits)
    sd_test_score = np.around(np.std(scores_test), decimals=score_digits)
    scores = pd.DataFrame({
        "train": [mean_train_score, sd_train_score], 
        "test": [mean_test_score, sd_test_score]}, 
        index=["mean","sd"])
    
    ### Pandas dataframe of feature average and SD importances
    mean_feature_importances = np.average(feature_importances, axis=0) 
    sd_feature_importances = np.std(feature_importances, axis=0)     
    original_feature_importances = pd.DataFrame(
        data={
        "feature": features.columns,
        "mean_feature_importance":mean_feature_importances, 
        "sd_feature_importance":sd_feature_importances})

    ### Confusion matrix
    # Pandas dataframe of sample true and predicted labels. 
    true_labels_as_list = [item for sublist in true_labels for item in sublist]           # flatten list
    predicted_labels_as_list = [item for sublist in predicted_labels for item in sublist] # flatten list 
    truth_prediction_df = pd.DataFrame({"truth":true_labels_as_list, "prediction":predicted_labels_as_list})
    # Computes the confusion matrix and returns a ConfusionMatrixDisplay ready to be plotted
    cm = confusion_matrix(
        y_true=truth_prediction_df.loc[:,"truth"] , 
        y_pred=truth_prediction_df.loc[:,"prediction"])
    confusion_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    ### Precision, recall, F-measure and support for each class.
    if positive_class == None:
        pos_label_scores = precision_recall_fscore_support(y_true = truth_prediction_df.loc[:,"truth"], y_pred=truth_prediction_df.loc[:,"prediction"])
    elif positive_class not in model.classes_:
        print("The 'positive_class' argument should be equal to one of the two sample classes. Check its value.")
    else:
        pos_label_scores = precision_recall_fscore_support(y_true = truth_prediction_df.loc[:,"truth"], y_pred=truth_prediction_df.loc[:,"prediction"])
    pos_label_scores = pd.DataFrame(pos_label_scores)

    print("This is the average training score: {:.3}".format(mean_train_score))    
    print("This is the average test score: {:.3}".format(mean_test_score))    
    
    return scores, original_feature_importances, confusion_matrix_disp, pos_label_scores


