import os
import unittest
import flask
from flask_testing import TestCase

import sys
sys.path.insert(0, '..')

#import model_api
from model_api import app, getUserResponse

class ModelTests( TestCase ):
    """
        Testing class for ML Model
    """
    def create_app(self):
        return app

    def setUp(self):
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        self.app = app.test_client()

    """
        TESTS
    """

    def test_app_is_up_and_running(self):
        #Sends request to app
        result = self.app.post('/')

        #Asserts the status of the response
        self.assertEqual(result.status_code, 200)


    """
    List of features:
        f1: Sepal Length
        f2: Sepal Width
        f3: Petal Length
        f4: Petal Width
    """

    #### CLASS 1: SETOSA ####
    def test_sestosa_class(self):
        response = self.app.get('/?f1=3&f2=4&f3=1.5&f4=0.2')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 0: Setosa")

    def test_sestosa_class2(self):
        response = self.app.get('/?f1=5&f2=&f3=&f4=')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 0: Setosa")

    #### CLASS 2: VERSICOLOUR ####
    def test_versicolor_class(self):
        response = self.app.get('/?f1=6&f2=2.8&f3=4.6&f4=1.3')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 1: Versicolour")


    def test_versicolor_class2(self):
        response = self.app.get('/?f1=6&f2=&f3=2&f4=1')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 1: Versicolour")

    #### CLASS 3: VIRGINICA ####
    def test_virginica_class(self):
        response = self.app.get('/?f1=6.4&f2=2.8&f3=5.6&f4=1.5')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 2: Virginica")

    def test_virginica_class2(self):
        response = self.app.get('/?f1=7&f2=&f3=6&f4=')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 2: Virginica")


if __name__ == "__main__":
    unittest.main()
