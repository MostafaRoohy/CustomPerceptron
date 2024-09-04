import numpy as np
import random


class CustomPerceptron:


    def __init__(self, epochs=10, learning_rate=1):

        self.epochs        = epochs
        self.learning_rate = learning_rate

        self.features = None
        self.labels   = None
        self.n        = None
        self.k        = None

        self.weights = None
        self.bias    = None

        self.errorsHistory = None
    #


    def fit(self, X, y):

        self.features = X
        self.labels   = y
        
        self.n = X.shape[0]
        self.k = X.shape[1]
        
        self.weights = np.ones(self.k)
        self.bias    = 0

        self.errorsHistory = np.zeros(self.epochs)



        for epoch in range(self.epochs):

            self.errorsHistory[epoch] = self.mean_error_whole(self.weights, self.bias, self.features, self.labels)

            i = random.randint(0, self.n-1)

            (self.weights, self.bias) = self.give_better_weights_bias_for_this_datapoint(self.weights, self.bias, self.features[i], self.labels[i])
        #
    #


    def error_this_datapoint(self, weights, bias, featuresDP, labelDP):

        guessThisDP = self.prediction_this_datapoint( weights, bias, featuresDP)

        if(guessThisDP==labelDP):

            return (0)
        #
        else:

            return (np.abs(np.dot(weights, bias, featuresDP)))
        #
    #


    def mean_error_whole(self, weights, bias):

        sum = 0

        for dp in range(self.n):

            sum += (self.error_this_datapoint(weights, bias, self.X[dp], self.y[dp]))
        #
    #


    def prediction_this_datapoint(self, weights, bias, featuresDP):

        scoreThisDP = np.dot(featuresDP, weights)+bias

        if(scoreThisDP>=0):

            return (1)
        #
        else:

            return (0)
        #
    #


    def give_better_weights_bias_for_this_datapoint(self, weights, bias, featuresDP, labelDP):

        guessThisDP = self.prediction_this_datapoint(weights, bias, featuresDP)


        if(guessThisDP==labelDP):

            return (weights, bias)
        #
        elif(guessThisDP==0 and labelDP==1):

            for i in range(self.k):

                weights[i] += (featuresDP[i]*self.learning_rate)
            #
            bias += (self.learning_rate)
        #
        elif(guessThisDP==1 and labelDP==0):

            for i in range(self.k):

                weights[i] -= (featuresDP[i]*self.learning_rate)
            #
            bias -= (self.learning_rate)
        #
    #
#