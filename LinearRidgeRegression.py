# Reference Source: https://github.com/llSourcell/linear_regression_live/blob/master/demo.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, average_precision_score

class LinearRegression:
    def __init__(self, filename):
        df = pd.read_csv(filename, header = None)
        self.X = np.array(df.drop([0], axis=1))
        self.y = np.array(df[0])

        self.learning_rate = 0.1
        self.num_iterations = 100
        self.cv_splits = 5
        self.l2_lambda = 1e-8
        self.division = 463715

    def hypothesis(self, b, W, X):
        return np.matmul(X, W) + b

    def compute_cost(self, b, W, X, y):
        total_cost = np.sum(np.square(y - self.hypothesis(b, W, X)))/(2*X.shape[0]) + self.l2_lambda*(np.dot(W, W) + b ** 2)
        return total_cost

    def gradient_descent_runner(self, X, y, b, W):
        cost_graph = []
        #For every iteration, optimize b, m and compute its cost
        for i in range(self.num_iterations):
            cost_graph.append(self.compute_cost(b, W, X, y))
            b, W = self.step_gradient(b, W, X, y)
        return [b, W, cost_graph]

    def step_gradient(self, b, W, X, y):
        #Calculate Gradient
        W_gradient = ((self.hypothesis(b, W, X) - y).dot(X))/X.shape[0] + self.l2_lambda*W
        b_gradient = (np.sum(X.dot(W) + b - y))/X.shape[0] + self.l2_lambda*b
        #Update current W and b
        W -= self.learning_rate * W_gradient
        b -= self.learning_rate * b_gradient
        #Return updated parameters
        return b, W

if __name__ == "__main__":
    lr = LinearRegression("YearPredictionMSD/YearPredictionMSD.txt")
    #X_train, X_test, y_train, y_test = train_test_split(lr.X, lr.y, test_size = 0.2, random_state = 1)

    # This split is provided by the repository. It avoids the 'producer effect' by making sure no song from a given artist ends up in both the train and test set.
    X_train, y_train = StandardScaler().fit_transform(lr.X[:lr.division]), lr.y[:lr.division]
    X_test, y_test = StandardScaler().fit_transform(lr.X[lr.division:]), lr.y[lr.division:]

    split_size = X_train.shape[0]//lr.cv_splits

    ev = []
    mae = []
    rmse = []
    msle = []
    r2 = []
    global_mae = []
    lambdas = []
    best_mae = 10
    best_l2 = 0
    b, W = None, None

    df = pd.DataFrame(np.concatenate((X_train,y_train[:, None]), axis = 1), columns = list(range(90, -1, -1)))
    df = shuffle(df)
    X_train = df.drop([0], axis = 1)
    y_train = df[0]

    for _ in range(8):
        ev = []
        mae = []
        rmse = []
        msle = []
        r2 = []
        print("Training and Testing for Lambda ", lr.l2_lambda)
        for i in range(lr.cv_splits):
            print("Cross Validation for Split ", i+1)
            start = i * split_size
            end = (i+1) * split_size
            X = np.concatenate((X_train[:start], X_train[end:]), axis = 0)
            y = np.concatenate((y_train[:start], y_train[end:]), axis=0)
            b = np.random.normal()
            W = np.random.normal(size=lr.X.shape[1])

            b, W, cost_graph = lr.gradient_descent_runner(X, y, b, W)

            plt.plot(range(lr.num_iterations), np.log(cost_graph))
            plt.title("Number of Iterations vs Cost")
            plt.show()

            X, y = X_train[start:end], y_train[start:end]
            h = lr.hypothesis(b, W, X)

            ev.append(explained_variance_score(y, h))
            print("Explained Variance : ", ev[-1])
            mae.append(mean_absolute_error(y, h))
            print("Mean Absolute Error : ", mae[-1])
            rmse.append(mean_squared_error(y, h) ** .5)
            print("Root Mean Squared Error : ", rmse[-1])
            msle.append(mean_squared_log_error(y, h))
            print("Mean Squared Log Error : ", msle[-1])
            r2.append(r2_score(y, h))
            print("R2 Score : ", r2[-1])

        global_mae.append(np.average(mae))
        lambdas.append(lr.l2_lambda)
        if best_mae > global_mae[-1]:
            best_mae = global_mae[-1]
            best_l2 = lr.l2_lambda
        lr.l2_lambda *= 3

    print("Test Data")
    lr.l2_lambda = best_l2
    print("With best hyperparameter lambda ", lr.l2_lambda)
    b = np.random.normal(scale=1 / X_train.shape[1] ** .5)
    W = np.random.normal(scale=1 / X_train.shape[1] ** .5, size=X_train.shape[1])

    b, W, cost_graph = lr.gradient_descent_runner(X_train, y_train, b, W)

    np.save("LRRWeights.npy", np.append(W, b))

    h = lr.hypothesis(b, W, X_test)

    ev.append(explained_variance_score(y_test, h))
    print("Explained Variance : ", ev[-1])
    mae.append(mean_absolute_error(y_test, h))
    print("Mean Absolute Error : ", mae[-1])
    rmse.append(mean_squared_error(y_test, h) ** .5)
    print("Root Mean Squared Error : ", rmse[-1])
    msle.append(mean_squared_log_error(y_test, h))
    print("Mean Squared Log Error : ", msle[-1])
    r2.append(r2_score(y_test, h))
    print("R2 Score : ", r2[-1])

    plt.plot(np.log(lambdas), global_mae)
    plt.title("Ridge Regression")
    plt.xlabel("Log of Lambda")
    plt.ylabel("Mean Absolute Error")
    plt.show()





