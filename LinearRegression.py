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
        df = shuffle(df)
        self.X = np.array(df.drop([0], axis=1))
        self.X = StandardScaler().fit_transform(self.X)
        self.y = np.array(df[0])

        self.learning_rate = 0.1
        self.num_iterations = 100
        self.cv_splits = 5

    def hypothesis(self, b, W, X):
        return np.matmul(X, W) + b

    def compute_cost(self, b, W, X, y):
        total_cost = np.sum(np.square(y - self.hypothesis(b, W, X)))
        return total_cost/(2*X.shape[0])

    def gradient_descent_runner(self, X, y, b, W):
        cost_graph = []
        #For every iteration, optimize b, m and compute its cost
        for i in range(self.num_iterations):
            cost_graph.append(self.compute_cost(b, W, X, y))
            b, W = self.step_gradient(b, W, X, y)
        return [b, W, cost_graph]

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
    lr = LinearRegression("YearPredictionMSD/YearPredictionMSD.txt")
    X_train, X_test, y_train, y_test = train_test_split(lr.X, lr.y, test_size = 0.2, random_state = 1)
    split_size = X_train.shape[0]//lr.cv_splits
    ev = []
    mae = []
    rmse = []
    msle = []
    r2  =[]

    """
    b, W = None, None
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

    plt.plot(range(lr.cv_splits), ev, "bo")
    #plt.plot(range(lr.cv_splits), mae, "r+")
    #plt.plot(range(lr.cv_splits), rmse, "g--")
    #plt.plot(range(lr.cv_splits), msle, "b.")
    plt.plot(range(lr.cv_splits), r2, "g^")
    plt.title("Split vs Metrics")
    plt.show()
    """

    print("Test Data")
    b = np.random.normal(scale=1 / X_train.shape[1] ** .5)
    # can get the size by checking it in the gradient_descent_runner function
    W = np.random.normal(scale=1 / X_train.shape[1] ** .5, size=X_train.shape[1])

    b, W, cost_graph = lr.gradient_descent_runner(X_train, y_train, b, W)

    np.save("LRWeights.npy", np.append(W, b))

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





