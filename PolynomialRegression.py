# Reference Source: https://github.com/llSourcell/linear_regression_live/blob/master/demo.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, average_precision_score

class PolynomialRegression:
    def __init__(self, filename):
        df = pd.read_csv(filename, header = None)
        df = shuffle(df)
        self.X = np.array(df.drop([0], axis=1))
        self.X = StandardScaler().fit_transform(self.X)
        self.y = np.array(df[0])

        self.learning_rate = 0.0015
        self.num_iterations = 100
        self.cv_splits = 5
        self.num_epochs = 3
        self.batch_size = 128
        self.degree = 2
        self.division = 463715

    def hypothesis(self, b, W, X):
        return np.matmul(X, W) + b

    def compute_cost(self, b, W, X, y):
        total_cost = np.sum(np.square(y - self.hypothesis(b, W, X)))
        return total_cost/(2*X.shape[0])

    def gradient_descent_runner(self, X, y, b, W):
        # For every iteration, optimize b, m and compute its cost
        cost = []
        for e in range(self.num_epochs):
            cost_graph = []
            print("Epoch", e + 1)
            i = 0
            while i < X.shape[0]//self.batch_size:
                temp_x = self.convert_poly(X[i * self.batch_size: (i + 1) * self.batch_size])
                b, W = self.step_gradient(b, W, temp_x, y[i * self.batch_size: (i + 1) * self.batch_size])
                cost_graph.append(self.compute_cost(b, W, temp_x, y[i * self.batch_size: (i + 1) * self.batch_size]))
                i += 1
                if i % 600 == 0:
                    print("Iteration", i, "Cost", cost_graph[-1])
                    cost.append(cost_graph[-1])
            temp_x = self.convert_poly(X[i * self.batch_size:])
            b, W = self.step_gradient(b, W, temp_x, y[i * self.batch_size:])
            cost_graph.append(self.compute_cost(b, W, temp_x, y[i * self.batch_size:]))

            cost.append(cost_graph[-1])
            print("Cost", cost[-1])

        return [b, W, cost]

    def convert_poly(self, x):
        return StandardScaler().fit_transform(PolynomialFeatures(degree=self.degree).fit_transform(x))

    def step_gradient(self, b, W, X, y):
        #Calculate Gradient
        W_gradient = (self.hypothesis(b, W, X) - y).dot(X)/X.shape[0]
        b_gradient = np.sum(X.dot(W) + b - y)/X.shape[0]
        #Update current W and b
        W -= self.learning_rate * W_gradient
        b -= self.learning_rate * b_gradient
        #Return updated parameters
        return b, W

if __name__ == "__main__":
    pr = PolynomialRegression("YearPredictionMSD/YearPredictionMSD.txt")
    #X_train, X_test, y_train, y_test = train_test_split(pr.X, pr.y, test_size = 0.2, random_state = 1)

    # This split is provided by the repository. It avoids the 'producer effect' by making sure no song from a given artist ends up in both the train and test set.
    X_train, y_train = pr.X[:pr.division], pr.y[:pr.division]
    X_test, y_test = pr.X[pr.division:], pr.y[pr.division:]

    split_size = X_train.shape[0]//pr.cv_splits
    ev = []
    mae = []
    rmse = []
    msle = []
    r2  =[]
    """
    #Commented block of code of Cross Validation. Used Hold out validation instead due to large training time.
    for i in range(pr.cv_splits):
        print("Cross Validation for Split ", i+1)
        start = i * split_size
        end = (i+1) * split_size
        X = np.concatenate((X_train[:start], X_train[end:]), axis = 0)
        y = np.concatenate((y_train[:start], y_train[end:]), axis=0)
        b = np.random.normal()
        #can get the size by checking it in the gradient_descent_runner function
        temp = pr.convert_poly(X_train[0:2]) #to get the shape of the weight vector
        W = np.random.normal(scale=1 / X.shape[1] ** .5, size = temp.shape)

        b, W, cost_graph = pr.gradient_descent_runner(X, y, b, W)

        #plt.plot(range(cost_graph), np.log(cost_graph))
        #plt.title("Number of Epochs vs Cost")
        #plt.show()

        X, y = X_train[start:end], y_train[start:end]
        h = pr.hypothesis(b, W, pr.convert_poly(X))

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
    """
    print("Test Data")
    b = np.random.normal(scale=1 / X_train.shape[1] ** .5)
    # can get the size by checking it in the gradient_descent_runner function
    temp = pr.convert_poly(X_train[0:2]) #to get the shape of the weight vector
    W = np.random.normal(scale=1 / X_train.shape[1] ** .5, size=temp.shape[1])

    b, W, cost_graph = pr.gradient_descent_runner(X_train, y_train, b, W)

    np.save("PRWeights.npy", np.append(W, b))

    temp_x = pr.convert_poly(X_test)
    h = pr.hypothesis(b, W, temp_x)

    plt.plot(range(len(cost_graph)), np.log(cost_graph))
    plt.title("Polynomial Regression")
    plt.xlabel("Iterations (x600)")
    plt.ylabel("Log of Cost")
    plt.show()

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





