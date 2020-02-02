# PluralSight Machine Learning Engineer Intern Coding Assessment
# Written By: Dominic Ridley

import sklearn
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics
import numpy as np
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
            model: naive bayes model fitted to the training set
        """
        X = self.X
        y = self.y

        k_fold = KFold(n_splits)
        model = self.model

        for train, test in k_fold.split(X):
            model.fit(X[train], y[train])
            p = model.predict( X[test] )
            # Add line for scores

        return model #return scores here?

    def predict(self, model, arg):
        """
        Returns prediction of user input based on new trained model

        Inputs:
            model: trained model
            args: user Inputs
        Outputs:
            prediction: A single value array with class label
        """
        prediction = model.predict(arg)

        return prediction

    #def getAccuracyScore(self, n_splits):
        """
        Gives an cross-validated accuracy score for the new model.

        Inputs:
            n_splits: number of sets to split the data into

        Returns:
            score: the accuracy score of the model.
        """
    #    score = cross_val_score(model, self.X, self.y, cv=n_splits)

    #    return score

if __name__ == "__main__":
    #Creates instance of NB model class
    m = NaiveBayesModel().kFoldCrossValidation(5)

    #Saves model
    pickle.dump(m, open("nb_model.p", "wb"))
