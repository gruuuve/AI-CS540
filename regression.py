import numpy as np
import math
from matplotlib import pyplot as plt

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_dataset(filename):
    """
    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    with open(filename) as file:
        num_cols = len(file.readline().split(','))
    dataset = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=range(1,num_cols), dtype=float)
    return dataset

def print_stats(dataset, col):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """ 
    # get number of data points
    column = dataset[:,col] # column of data to look at
    num_points = dataset.shape[0]
    print(num_points)
    # get mean of column
    mean = 0
    for item in column: mean += item
    mean = mean / num_points
    print('{:.2f}'.format(mean))
    # get standard deviation for column
    std_dev = 0
    for item in column: std_dev += (item - mean) ** 2
    std_dev = (std_dev / (num_points - 1)) ** 0.5
    print('{:.2f}'.format(std_dev))
    pass

def regression(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    # setup function f(x)
    # get columns for beta
    col_set = []
    col_set = dataset[:, cols] 
    # iterate through summing the rows w/ betas
    mse = 0
    for row in range(dataset.shape[0]):
        mse += (betas[0] + sum(col_set[row]*betas[1:]) - dataset[row][0]) ** 2
    mse = mse / dataset.shape[0]
    return mse

def gradient_descent(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    #similar to method above, calculation of function is different
    col_set = []
    grads = []
    col_set = dataset[:, cols]
    n = dataset.shape[0]
    for val in range(len(betas)): # iterate for each beta
        sums = 0
        for row in range(n): # sum function
            if val == 0: # first beta don't multiply with x_i
                sums += (betas[0] + sum(col_set[row] * betas[1:]) - dataset[row][0])
            else:
                sums += (betas[0] + sum(col_set[row] * betas[1:]) - dataset[row][0]) * col_set[row][val-1]
        sums = sums * 2 / n # final part of sum calc
        grads.append(sums)
    return np.array(grads)

def iterate_gradient(dataset, cols, betas, T, eta):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    mse = 0
    grads = [] # store gradient descent
    old_betas = []
    new_betas = [] # store new beta values
    old_betas = betas # just for first iteration, changed after that
    for i in range(T + 1): # iterate for T + 1 times, extra loop for initial data
        mse = regression(dataset, cols, old_betas) # get current mse
        grads = gradient_descent(dataset, cols, old_betas) # get gradient descent
        new_betas = old_betas - eta*grads
        # print out results
        if i >= 1:
            print('{} {:.2f}'.format(i, mse), end = ' ') #, mse, old_betas)
            for j in range(len(old_betas)):
                print('{:.2f}'.format(old_betas[j]), end = ' ')
            print() # go to next line
        old_betas = new_betas
    pass

def compute_betas(dataset, cols):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    betas = [] # return values
    mse = 0
    x = np.ones((dataset.shape[0], 1))
    #for i in range(cols):
    col_set = []
    col_set = np.array(dataset[:, cols])
    #col_set.reshape(col_set.shape[0],-1)
    x = np.hstack((x, col_set)) # create array with col 0 = 1, rest features
    betas = np.matmul( np.matmul( np.linalg.inv(np.matmul( np.transpose(x),x )), np.transpose(x) ), dataset[:, 0] )
    mse = regression(dataset, cols, betas)
    return (mse, *betas)

def predict(dataset, cols, features):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    result = 0
    # use dataset and cols to get betas, then multiply features by betas to get result
    betas = compute_betas(dataset, cols) # get set of betas
    result = betas[1] + sum(np.array(betas[2:]) * np.array(features)) # index + 1 to skip mse in betas
    return result

def synthetic_datasets(betas, alphas, X, sigma):
    """
    INPUT:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    n = len(X) # number of data points
    lin = np.zeros((n, 2)) # arrays for linear synthetic data
    quad = np.zeros((n, 2)) # quadratic synthetic data
    X = np.array(X)

    # linear data
    z_l = np.random.normal(0, sigma, X.shape) # array of z values to add to syndata
    z_q = np.random.normal(0, sigma, X.shape)
    # fix for types
    #X = np.array(X).reshape(n,1)
    l_int = [elem * betas[1] for elem in X]
    q_int = [alphas[1] * (elem**2) for elem in X]
    #lin[:,0] = betas[0] + x_int + z_l
    #lin[:,1] = X
    for i in range(n):
        lin[i,0] = betas[0] + l_int[i] + z_l[i]
        lin[i,1] = X[i]
        quad[:,0] = alphas[0] + q_int[i] + z_q[i]
        quad[:,1] = X[i]
    return lin, quad

def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')
    # TODO: Generate datasets and plot an MSE-sigma graph
    X = np.array(np.zeros((1000, 1)))
    lin = []
    quad = []
    mse_lin = np.array(np.zeros((10, 1)))
    mse_quad = np.array(np.zeros((10, 1)))
    for i in range(1000): # generate a size 1000x1 array w/ random floats b/w -100, 100
        X[i] = float(np.random.uniform(-100, 100))
    betas = (np.random.uniform(0.001, 10), np.random.uniform(0.001, 10)) # generate couples
    alphas = (np.random.uniform(0.001, 10), np.random.uniform(0.001, 10))
    sigmas = [10e-4, 10e-3, 10e-2, 10e-1, 1, 10, 10e2, 10e3, 10e4, 10e5]
    for i in range(len(sigmas)): # get synthetic datasets and corresponding MSEs
        lin, quad = synthetic_datasets(betas, alphas, X, sigmas[i])
        mse_lin[i] = compute_betas(np.array(lin), cols=[1])[0]
        mse_quad[i] = compute_betas(np.array(quad), cols=[1])[0]
    fig = plt.figure()
    plt.xlabel('sigma')
    plt.xscale("log")
    plt.ylabel('MSEs')
    plt.yscale("log")
    plt.plot(sigmas, mse_lin, "-o", label = "Linear MSEs")
    plt.plot(sigmas, mse_quad,"-o", label = "Quadratic MSEs")
    plt.legend()
    fig.savefig("mse.pdf")

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
