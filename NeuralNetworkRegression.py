#Reference Source: https://github.com/llSourcell/Make_a_neural_network/blob/master/demo.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, average_precision_score

class NeuralNetworkRegression:
    def __init__(self, filename):
        df = pd.read_csv(filename, header = None)
        self.X = np.array(df.drop([0], axis=1))
        self.y = np.array(df[0])

        self.learning_rate = 0.001
        self.num_iterations = 100
        self.batch_size = 512
        self.cv_splits = 5
        self.n_hidden = 30
        self.num_epochs = 300
        self.division = 463715
        self.weights_input_hidden = None#np.random.normal(scale=1 / self.X.shape[1] ** .5, size=(self.X.shape[1], self.n_hidden))
        self.weights_hidden_output = None#np.random.normal(scale=1 / self.X.shape[1] ** .5, size=self.n_hidden)

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def cal_grad(self, X, Y):
        hidden_input = np.dot(X, self.weights_input_hidden)
        hidden_output = self.sigmoid(hidden_input)
        output = np.dot(hidden_output, self.weights_hidden_output)

        error = Y - output
        output_error_term = error

        hidden_error = np.outer(output_error_term, self.weights_hidden_output)
        hidden_error_term = hidden_error * hidden_output * (1. - hidden_output)

        del_w_hidden_output = np.dot(hidden_output.T, output_error_term)
        del_w_input_hidden = np.matmul(X.T, hidden_error_term)
        return [del_w_input_hidden, del_w_hidden_output]

    def train(self, X, Y):
        last_loss = None
        cost_graph = []
        for e in range(self.num_epochs):
            i = 0
            while i < X.shape[0]//self.batch_size:
                dwih, dwho = self.cal_grad(X[i*self.batch_size : (i+1)*self.batch_size], Y[i*self.batch_size : (i+1)*self.batch_size])

                self.weights_input_hidden += self.learning_rate * dwih / self.batch_size
                self.weights_hidden_output += self.learning_rate * dwho / self.batch_size

                i += 1

            dwih, dwho = self.cal_grad(X[i * self.batch_size:], Y[i * self.batch_size:])
            remaining_batch_size = X.shape[0] - i * self.batch_size

            self.weights_input_hidden += self.learning_rate * dwih / remaining_batch_size
            self.weights_hidden_output += self.learning_rate * dwho / remaining_batch_size

            # Printing out the mean square error on the training set
            if e % (self.num_epochs / 10) == 0:
                hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden))
                out = np.dot(hidden_output, self.weights_hidden_output)
                loss = np.mean((out - Y) ** 2)
                cost_graph.append(loss)
                if last_loss and last_loss < loss:
                    print("Train loss: ", loss, "  WARNING - Loss Increasing")
                else:
                    print("Train loss: ", loss)
                last_loss = loss
        return cost_graph

if __name__ == "__main__":
    nnr = NeuralNetworkRegression("YearPredictionMSD/YearPredictionMSD.txt")
    #X_train, X_test, y_train, y_test = train_test_split(nnr.X, nnr.y, test_size = 0.2, random_state = 1)

    #This split is provided by the repository. It avoids the 'producer effect' by making sure no song from a given artist ends up in both the train and test set.
    X_train, y_train = StandardScaler().fit_transform(nnr.X[:nnr.division]), nnr.y[:nnr.division]
    X_test, y_test = StandardScaler().fit_transform(nnr.X[nnr.division:]), nnr.y[nnr.division:]

    split_size = X_train.shape[0]//nnr.cv_splits
    ev = []
    mae = []
    rmse = []
    msle = []
    r2  =[]
    hidden_nodes = []
    global_mae = []

    X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    for i in range(5):
        nnr.n_hidden = (i + 3)*30
        hidden_nodes.append(nnr.n_hidden)
        """
        #Commented block of code for K-Fold Cross Validation. Using Hold out Cross Validation instead.
        for i in range(nnr.cv_splits):
            print("Cross Validation for Split ", i+1)
            start = i * split_size
            end = (i+1) * split_size
            X = np.concatenate((X_train[:start], X_train[end:]), axis = 0)
            y = np.concatenate((y_train[:start], y_train[end:]), axis = 0)

            nnr.weights_input_hidden = np.random.normal(scale=1 / X.shape[1] ** .5, size=(X.shape[1], nnr.n_hidden))
            nnr.weights_hidden_output = np.random.normal(scale=1 / X.shape[1] ** .5, size=nnr.n_hidden)
    
            cost_graph = nnr.train(X, y)
    
            plt.plot(range(len(cost_graph)), np.log(cost_graph))
            plt.title("Number of Iterations vs Cost")
            plt.show()
    
            X, y = X_train[start:end], y_train[start:end]
            hidden_output = nnr.sigmoid(np.dot(X, nnr.weights_input_hidden))
            h = np.dot(hidden_output, nnr.weights_hidden_output)
    
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
        print(nnr.n_hidden, "hidden nodes")
        print("Hold out Crosss Validation")

        nnr.weights_input_hidden = np.random.normal(scale=1 / X_t.shape[1] ** .5, size=(X_t.shape[1], nnr.n_hidden))
        nnr.weights_hidden_output = np.random.normal(scale=1 / X_t.shape[1] ** .5, size=nnr.n_hidden)

        cost_graph = nnr.train(X_t, y_t)

        hidden_output = nnr.sigmoid(np.dot(X_val, nnr.weights_input_hidden))
        h = np.dot(hidden_output, nnr.weights_hidden_output)

        ev.append(explained_variance_score(y_val, h))
        print("Explained Variance : ", ev[-1])
        global_mae.append(mean_absolute_error(y_val, h))
        print("Mean Absolute Error : ", global_mae[-1])
        rmse.append(mean_squared_error(y_val, h) ** .5)
        print("Root Mean Squared Error : ", rmse[-1])
        msle.append(mean_squared_log_error(y_val, h))
        print("Mean Squared Log Error : ", msle[-1])
        r2.append(r2_score(y_val, h))
        print("R2 Score : ", r2[-1])

    plt.plot(hidden_nodes, global_mae)
    plt.title("Neural Network Regression")
    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Mean Absolute Error")
    plt.show()

    print("Test Data")

    nnr.weights_input_hidden = np.random.normal(scale=1 / X_train.shape[1] ** .5, size=(X_train.shape[1], nnr.n_hidden))
    nnr.weights_hidden_output = np.random.normal(scale=1 / X_train.shape[1] ** .5, size=nnr.n_hidden)

    cost_graph = nnr.train(X_train, y_train)

    np.save("NRWeightsIH.npy", nnr.weights_input_hidden)
    np.save("NRWeightsHO.npy", nnr.weights_hidden_output)

    hidden_output = nnr.sigmoid(np.dot(X_test, nnr.weights_input_hidden))
    h = np.dot(hidden_output, nnr.weights_hidden_output)

    ev.append(explained_variance_score(y_test, h))
    print("Explained Variance : ", ev[-1])
    global_mae.append(mean_absolute_error(y_test, h))
    print("Mean Absolute Error : ", global_mae[-1])
    rmse.append(mean_squared_error(y_test, h) ** .5)
    print("Root Mean Squared Error : ", rmse[-1])
    msle.append(mean_squared_log_error(y_test, h))
    print("Mean Squared Log Error : ", msle[-1])
    r2.append(r2_score(y_test, h))
    print("R2 Score : ", r2[-1])


