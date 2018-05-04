# Reference Source: https://github.com/llSourcell/linear_regression_live/blob/master/demo.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, average_precision_score
from mpl_toolkits.mplot3d import Axes3D

class PolynomialRegression:
    def __init__(self, filename):
        df = pd.read_csv(filename, header = None)
        self.X = np.array(df.drop([0], axis=1))
        self.y = np.array(df[0])

        self.learning_rate = 0.0015
        self.num_iterations = 100
        self.cv_splits = 5
        self.num_epochs = 3
        self.batch_size = 128
        self.degree = 2
        self.l1_lambda = 0.00001
        self.l2_lambda = 0.00001
        self.division = 463715

    def hypothesis(self, b, W, X):
        return np.matmul(X, W) + b

    def compute_cost(self, b, W, X, y):
        total_cost = np.sum(np.square(y - self.hypothesis(b, W, X))) + self.l2_lambda * (np.dot(W, W) + b ** 2) + self.l1_lambda * (np.sum(np.fabs(W)) + abs(b))
        return total_cost/(2*X.shape[0])

    def gradient_descent_runner(self, X, y, b, W):
        cost = []
        for e in range(self.num_epochs):
            cost_graph = []
            print("Epoch", e + 1)
            i = 0
            while i < X.shape[0] // self.batch_size:
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
            print("Iteration", i, "Cost", cost_graph[-1])
            print("Cost", cost[-1])

        return [b, W, cost]

    def convert_poly(self, x):
        return StandardScaler().fit_transform(PolynomialFeatures(degree=self.degree).fit_transform(x))

    def step_gradient(self, b, W, X, y):
        #Calculate Gradient
        W_gradient = ((self.hypothesis(b, W, X) - y).dot(X) + self.l2_lambda*W + self.l1_lambda)/X.shape[0]
        b_gradient = (np.sum(X.dot(W) + b - y) + self.l2_lambda*b + self.l1_lambda)/X.shape[0]
        #Update current W and b
        W -= self.learning_rate * W_gradient
        b -= self.learning_rate * b_gradient
        #Return updated parameters
        return b, W

if __name__ == "__main__":
    pr = PolynomialRegression("YearPredictionMSD/YearPredictionMSD.txt")
    #X_train, X_test, y_train, y_test = train_test_split(pr.X, pr.y, test_size = 0.2, random_state = 1)

    # This split is provided by the repository. It avoids the 'producer effect' by making sure no song from a given artist ends up in both the train and test set.
    X_train, y_train = StandardScaler().fit_transform(pr.X[:pr.division]), pr.y[:pr.division]
    X_test, y_test = StandardScaler().fit_transform(pr.X[pr.division:]), pr.y[pr.division:]

    split_size = X_train.shape[0]//pr.cv_splits
    ev = []
    mae = []
    rmse = []
    msle = []
    r2  =[]
    global_mae = []

    best_mae = 10
    best_l2 = 0
    best_l1 = 0

    # Hold out validation instead of Cross Validation because of large training time

    X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    lambdas1 = []
    lambdas2 = []
    for _ in range(3):
        pr.l2_lambda = 0.00001
        for _ in range(3):
            print("Training for Lambda L1", pr.l1_lambda, "and L2", pr.l2_lambda)
            temp = pr.convert_poly(X_t[0:2])  # to get the shape of the weight vector
            b = np.random.normal(scale=1 / temp.shape[1] ** .5)
            W = np.random.normal(scale=1 / temp.shape[1] ** .5, size=temp.shape[1])
            b, W, cost_graph = pr.gradient_descent_runner(X_t, y_t, b, W)

            print("Testing on Validation dataset")
            h = pr.hypothesis(b, W, pr.convert_poly(X_val))

            print("Explained Variance : ", explained_variance_score(y_val, h))
            global_mae.append(mean_absolute_error(y_val, h))
            print("Mean Absolute Error : ", global_mae[-1])
            print("Root Mean Squared Error : ", mean_squared_error(y_val, h) ** .5)
            print("Mean Squared Log Error : ", mean_squared_log_error(y_val, h))
            print("R2 Score : ", r2_score(y_val, h))
            """
            #Commented block for Cross Validation. Used Hold out validation due to large training time.
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
    
                plt.plot(range(cost_graph), np.log(cost_graph))
                plt.title("Number of Epochs vs Cost")
                plt.show()
    
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
            lambdas2.append(pr.l2_lambda)
            lambdas1.append(pr.l1_lambda)
            if best_mae > global_mae[-1]:
                best_mae = global_mae[-1]
                best_l2 = pr.l2_lambda
                best_l1 = pr.l1_lambda
            pr.l2_lambda *= 3
        pr.l1_lambda *= 3

    print("Test Data")
    pr.l2_lambda = best_l2
    pr.l1_lambda = best_l1
    print("With best hyperparameter lambda L1", pr.l1_lambda, "and L2", pr.l2_lambda)
    temp = pr.convert_poly(X_t[0:2])  # to get the shape of the weight vector
    b = np.random.normal(scale=1 / temp.shape[1] ** .5)
    W = np.random.normal(scale=1 / temp.shape[1] ** .5, size=temp.shape[1])

    b, W, cost_graph = pr.gradient_descent_runner(X_train, y_train, b, W)

    np.save("PRRWeights.npy", np.append(W, b))

    temp_x = pr.convert_poly(X_test)
    h = pr.hypothesis(b, W, temp_x)

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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(np.log(lambdas1, order=3), np.log(lambdas2, order=3), global_mae)
    ax.text2D(0.00, 1.00, "Elastic Net Regression", transform=ax.transAxes)
    ax.set_xlabel("Log of Lambda L1")
    ax.set_ylabel("Log of Lambda L2")
    ax.set_zlabel("Mean Absolute Error")
    plt.show()



