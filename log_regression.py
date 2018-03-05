import numpy as np
import matplotlib.pyplot as plt


def sig(a):
    s = 1 / (1 + np.exp(-a))
    return s


def grad_cost(w, b, X, Y):
    m = X.shape[1]

    A = sig(np.dot(w.T, X) + b)

    J = - 1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dz = (1 / m) * (A - Y)

    dw = np.dot(X, dz.T)
    db = np.sum(dz)

    cost = np.squeeze(J)
    grad = {"dw": dw, "db": db}

    return grad, cost


def model(Xtrain, Ytrain, iterations, learning_rate):
    b = 0
    w = np.zeros(shape=(Xtrain.shape[0], 1))
    costs = []

    for i in range(iterations):
        # m = Xtrain.shape[1]
        grad, cost = grad_cost(w, b, Xtrain, Ytrain)
        if (i % 5000 == 0):
            costs.append(cost)
            print("At iteration %i = %f" % (i, cost))
        w = w - learning_rate * grad["dw"]
        b = b - learning_rate * grad["db"]

    params = {"w": w, "b": b}

    return params, grad, costs


# def prediction(w, b, Xtest):
#     m = Xtest.shape[1]
#     Y_prediction = np.zeros((1,m))


def main():
    np.random.seed(41)
    num_observations = 100

    X_train = np.random.random_integers(0, 100, (3, 100))
    Y_train = np.random.random_integers(0, 1, (1, 100))

    X_test = np.random.random_integers(0, 100, (3, 10))
    Y_test = np.random.random_integers(0, 1, (1, 10))

    parameters, grads, costs = model(X_train, Y_train, 100000, 0.0005)

    w = parameters["w"]
    b = parameters["b"]

    print(w)
    print(b)
    # Y_prediction = predict(w,b,Xtest)


if __name__ == "__main__":
    main()
