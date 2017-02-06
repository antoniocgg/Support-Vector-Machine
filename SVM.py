import parser
import numpy as np
import sys
import copy


'''
Note: if a parameter is denoted as an array, it is an np.array
'''


    
''' Returns 1 or -1 as our class guess
    
    Params: weights: weight array with bias as weights[0]
            point:   data point array with true class as x_n[0]
    '''
def get_sign(weights, point):
    # remove class from the line, and bias from weights when calculating
    sign = weights[0] + sum(np.array(point[1:])*weights[1:]) # vector multiplication
    return 1 if sign >= 0.0 else -1


''' Returns: gradients for weights and bias
                 wgradient: weight gradient array
                 bgradient: bias gradient float

    Params: weights: weight array with bias as weights[0]
            x_n:     data point array with true class as x_n[0]
            N:       number of training points in training data
            C:       arbitrary capacity
    '''
def get_gradients(weights, x_n, N, C=.05):
    bias = weights[0]
    y_n = x_n[0]
    wgradient = bgradient = 0
    if (y_n * (sum(weights[1:]*x_n[1:])+bias)) < 1:
        # incorrect guess
        wgradient = (1/N)*(weights[1:]) - (C*y_n*x_n[1:])
        bgradient = (-C)*(y_n)
    else:
        # correct guess
        wgradient = (1/N)*(weights[1:])
    return wgradient, bgradient


''' Returns: trained weights in np.array

    Params: train_data:    training data
            learning_rate: learning rate
            epochs:        number of times to loop through training data
            C:             arbitrary capacity
    '''
def train_weights(train_data, learning_rate, epochs, C=.05):
    weight_length = len(train_data[0])
    weights = np.array([0] * weight_length).astype(float)
    N = len(train_data)
    for iteration in range(epochs):  # loop epochs
        for point in train_data: # [0.2, 0.4, ..., 1]
            wgradient, bgradient = get_gradients(weights, point, N, C)   # gradients
            # update is below
            weights[0] -= learning_rate*bgradient    # bias
            weights[1:] -= learning_rate*wgradient
    return weights


''' Returns: accuracy

    Params: test_data: data to test
            weights:   weight array
    '''
def svm(test_data, weights):
    right = 0
    wrong = 0
    for line in test_data:
        true_class = line[0]
        estimated_class = get_sign(weights, line)
        if estimated_class == true_class:
            right += 1
        else:
            wrong += 1
    accuracy = float(right)/float((right+wrong))
    return accuracy


''' Returns: none; prints accuracy to console

    Action: loads and parses datasets, gets weights, tests test data
    '''
def main():#learning_rate, epochs, C):
    # load datasets
    file_train = "a7a.train"
    # file_test = sys.argv[1]
    file_test = "a7a.dev"

    # parse datasets
    train_data = parser.parse(file_train)
    test_data = parser.parse(file_test)

    # get weight array
    
    epochs = 4
    learning_rate = 0.45
    C = 0.0127

    print "Training..."
    weights = train_weights(train_data, learning_rate, epochs, C)
    
    print "Weights are trained " + str(epochs) + " times, with a learning rate of " \
          + str(learning_rate) + " and a capacity of " + str(C)
    print "Testing..."
    print "Tested on " + str(file_test) + ", accuracy is " + \
          str(svm(test_data, weights)*100) + "%"


if __name__ == '__main__':
    main()


# best is 0.45 lr 0.0127 c 4 epochs with accuracy of 0.851125
