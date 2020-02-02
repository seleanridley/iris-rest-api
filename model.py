# PluralSight Machine Learning Engineer Intern Coding Assessment
# Written By: Dominic Ridley

from flask import Flask
import requests
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics
import pickle

#Loads Iris Dataset and declares variable
iris = datasets.load_iris()


class NaiveBayesModel:
    def __init__(self):
        """
        Defines the machine learning model class

        Defines:
            model: defines Gaussian Naives Bayes model
            X: iris data/features
            y: class labels

        """
        self.model = GaussianNB();
        self.X = iris.data
        self.y = iris.target

    def kFoldCrossValidation(self, n_splits ):
        """
        Splits the data into training and testing sets.
        Implements K-fold Cross Validation using the number of splits passed
        to the function.

        Inputs:
            n-splits: number of sets to split the data into

        Returns:
            model_fit: naive bayes model fitted to the training set
            predict: cross-validation predictions
        """
        X = self.X
        y = self.y

        k_fold = KFold(n_splits)

        for train_index, test_index in k_fold.split(iris_X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


        model_fit = self.model.fit(X_train, y_train)
        predict = cross_val_predict(model, iris_X, iris_y, cv=n_splits)
        score = self.getAccuracyScore( model, iris_X, iris_y, n_splits)


        return model_fit, predict

    def getAccuracyScore(self, n_splits):
        """
        Gives an cross-validated accuracy score for the new model.

        Inputs:
            n_splits: number of sets to split the data into

        Returns:
            score: the accuracy score of the model.
        """
        model, predict = kFoldCrossValidation(n_splits)
        score = cross_val_score(model, self.X, self.y, cv=n_splits)

        return score

if __name__ == "__main__":
    #Creates instance of NB model class
    m = NaiveBayesModel(iris_X, iris_y)

    #Saves
    #pickle.dump([model, pre], open("nb_model.p", "wb"))
