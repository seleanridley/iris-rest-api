# Iris Rest API

Coding Assessment Submission for Pluralsight Machine Learning Engineer Internship

Dominic Ridley

## Prerequisites

Dependencies:
```
pip install flask flask_restful scikit-learn numpy
```

## Running the Model

The API is run through a flask template that is called in model_api.py. To run, execute in the root directory:
```
python3 model_api.py
```
The model can be accessed at https:/localhost/9999/

## Questions

1. When handling datasets that do not easily fit into system memory, I would break down the data into different batches for testing and running, as well as the logical tasks of testing into separate testing jobs that can be run and validated one at a time or on different clusters at once.

2. My optimal versioning strategy would be to version through using a custom request header.
```
  Version: MAJOR.MINOR.PATCH
```
 Training a model on new data shouldn't take away from the functionality of the model, except for its accuracy. Applications that depend on the model won't be broken by the new data. Therefore, it would add an increment to the minor version number.
 
 Pros:
  - It doesn't create a busy URL and allows all versions to be accessed from the same resource address. No rerouting.
  - Doesn't have to mix query parameters with control parameters.

Cons:
  - It may not be very accessible to a client that isn't familiar with the developer side.
  - Changes to the header may affect if/how the version number is received.


3. I decided to use a Naive Bayes classifier based on Bayes Theorem, which describes the probability of an event given prior knowledge to conditions pertaining to the event. Since this is a supervised learning problem, the class labels are known which allow easy calculation of the class prior probabilities and the likelihood function. A Bayes classifier was one of my first considerations for building a classifier for the Iris data. 

Benefits:
 - It performs well for multi-class predictions.
 - It works for training on small datasets.
 - Training time has the order O(N) which will scale well as the size of the data increases.
 
Drawbacks:
 - If a feature has zero frequency, the total probability is zero.
 - If a feature is missing, it can require a default value, chosen to be 0 in my model, which can affect accuracy.
 - Assumes class features are independent.
 - The hypothesis function is linear so it may underfit the data if it grows in complexity.

