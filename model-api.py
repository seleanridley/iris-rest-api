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
        u_inputs = [[]]

        trained_model = model.kFoldCrossValidation(5)

        #Splits the user input by whitespace
        p = request.args['user_in'].split()
        for x in p:
            u_inputs[0].append( float(x) ) #Coverts to floats

        prediction = model.predict(trained_model, u_inputs )

        return flask.render_template('api.html',
                                      user_in=input,
                                      prediction=prediction[0])
    else:
        return flask.render_template('api.html',
                                      user_in=input,
                                      prediction='')
