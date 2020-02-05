# PluralSight Machine Learning Engineer Intern Coding Assessment
# REST API
# Written By: Dominic Ridley


import flask
from model import NaiveBayesModel
import pickle
import numpy as np
from flask_restful import reqparse, abort, Api, Resource
from model import NaiveBayesModel
from flask import jsonify, request

app = flask.Flask(__name__, template_folder='templates')
api = Api(app)

model = NaiveBayesModel()

@app.route("/", methods=["GET","POST"])
def getPredictions():
    """
    POSTs predictions to html template after GETing user Inputs
    """
    if(request.args):
        u_inputs = []

        #Calls trained model
        trained_model = model.kFoldCrossValidation(5)

        for i in range(1,5):

            c_feature = 'f' + str(i)

            #Checks empty input
            if len(request.args[c_feature]) == 0:
                u_inputs.append( 0 )
            #Checks if input is negative
            elif float(request.args[c_feature]) < 0:
                raise ValueError("Inputs less than 0 are not accepted")
            #Pads feature input list with zeros for empty inputs
            else:
                u_inputs.append( float( request.args[c_feature] ) )

        prediction = model.predict(trained_model, [u_inputs] )

        outputs = { 0: 'Class 0: Setosa',
                    1: 'Class 1: Versicolour',
                    2: 'Class 2: Virginica'}

        return flask.render_template('api.html',
                                      f1=input,
                                      f2=input,
                                      f3=input,
                                      f4=input,
                                      prediction=outputs.get(prediction[0]))
    else:
        return flask.render_template('api.html',
                                      f1=input,
                                      f2=input,
                                      f3=input,
                                      f4=input,
                                      prediction='')

#Helper Methods
def getUserResponse(feature1, feature2, feature3, feature4):

    return app.post('/', data=dict(f1=feature1,
                                   f2=feature2,
                                   f3=feature3,
                                   f4=feature4),
                                   follow_redirects=True)


if __name__ == "__main__":
    app.run("localhost", "9999", debug=True)
