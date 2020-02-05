import os
import unittest
import flask
from flask_testing import TestCase

import sys
sys.path.insert(0, '..')

#import model_api
from model_api import app, getUserResponse

class ModelApiTests( TestCase ):
    """
        Testing class for ML Model API
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


    def test_empty(self):
        response = self.app.get('/')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "")

    def test_1_feature(self):
        response = self.app.get('/?f1=1&f2=&f3=&f4=')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 2: Virginica")

    def test_2_features(self):
        response = self.app.get('/?f1=1&f2=2&f3=&f4=')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 2: Virginica")

    def test_3_features(self):
        response = self.app.get('/?f1=1&f2=2&f3=3&f4=')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 2: Virginica")

    def test_4_features(self):
        response = self.app.get('/?f1=0.1&f2=0.2&f3=0.7&f4=0.5')
        self.assert_template_used('api.html')
        self.assert_context("prediction", "Class 2: Virginica")

    def test_string_input(self):
        with self.assertRaises(ValueError):
            response = self.app.get('/?f1=t&f2=2&f3=3&f4=4')

    def test_character_input(self):
        with self.assertRaises(ValueError):
            response = self.app.get('/?f1=^&f2=2&f3=3&f4=4')

    def test_negative_inputs(self):
        with self.assertRaises(ValueError):
            response = self.app.get('/?f1=-1&f2=2&f3=3&f4=4')


if __name__ == "__main__":
    unittest.main()
