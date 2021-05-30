import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def grad_update(g_x, x, step):
    return x - step * g_x(x)


def s_grad_update(g, x, step):
    return x - step * g


def gradient_method(f, gf, x0, epsilon=1e-10):
    x = x0
    xt = []
    fs = []
    gval = gf(x)
    iter = 0
    while np.linalg.norm(gval) >= epsilon:
        step = 0.1
        x = grad_update(gf, x, step)
        xt.append(x)
        # print('iter= {:2d} f(x)={:10.10f} x={}'.format(iter, f(x), x))
        # print(ts[iter])
        gval = gf(x)
        fs.append(f(x))
        iter += 1

    return x, xt, fs


def svm_with_sgd(x, y, lam, epochs, l_rate, sgd_type):
    np.random.seed(2)
    m = len(x)
    d = len(x[0])
    w = np.random.rand(d)
    b = np.random.rand()
    if sgd_type == 'theory':
        bt = []
        wt = []
        for i in range(m * epochs):
            rand_index = np.random.randint(m)
            sample = x[rand_index]
            label = y[rand_index]
            if 1 - label * (np.inner(w, sample) + b) <= 0:
                w_sub_garedient = l_rate * 2 * lam * w
                b_sub_garedient = 0
            else:
                w_sub_garedient = l_rate * (2 * lam * w - label * sample)
                b_sub_garedient = -l_rate * label
            b = s_grad_update(b_sub_garedient, b, l_rate)
            w = s_grad_update(w_sub_garedient, w, l_rate)
            bt.append(b)
            wt.append(w)
        return np.mean(wt, axis=0), np.mean(bt, axis=0)
    elif sgd_type == 'practical':
        w_sub_garedient = 0
        b_sub_garedient = 0
        for i in range(epochs):
            random_order = np.random.permutation(m)
            for j in random_order:
                sample = x[j]
                label = y[j]
                if (1 - label * ((np.dot(w, sample)) + b)) <= 0:
                    w_sub_garedient = 2 * lam * w
                    b_sub_garedient = 0
                else:
                    w_sub_garedient = 2 * lam * w - label * sample
                    b_sub_garedient = -1 * label
                b = s_grad_update(b_sub_garedient, b, l_rate)
                w = s_grad_update(w_sub_garedient, w, l_rate)
        return w, b


def svm_with_sgd2(x, y, lam, epochs, l_rate, sgd_type):
    np.random.seed(2)
    m = len(x)
    d = len(x[0])
    w = np.random.rand(d)
    b = np.random.rand()
    if sgd_type == 'theory':
        bt = np.random.uniform(size=m * epochs)
        wt = np.random.rand(m * epochs, d)
        for i in range(1,m * epochs):
            rand_index = np.random.randint(m)
            sample = x[rand_index]
            label = y[rand_index]
            if (1 - label * (np.dot(wt[i - 1], sample) + bt[i - 1])) <= 0:
                w_sub_garedient = l_rate* 2 * lam * wt[i]
                b_sub_garedient = 0
            else:
                w_sub_garedient = l_rate * (2 * lam * wt[i - 1] - label * sample)
                b_sub_garedient = -l_rate * label
            wt[i] = wt[i - 1] - w_sub_garedient
            bt[i] = bt[i - 1] - b_sub_garedient
        return np.mean(wt, axis=0), np.mean(bt, axis=0)

    elif sgd_type == 'practical':
        w_sub_garedient = 0
        b_sub_garedient = 0

        for i in range(epochs):
            random_order = np.random.permutation(m)
            for j in random_order:
                sample = x[j]
                label = y[j]
                if (1 - label * ((np.dot(w, sample)) + b)) <= 0:
                    w_sub_garedient = 2 * lam * w
                    b_sub_garedient = 0
                else:
                    w_sub_garedient = 2 * lam * w - label * sample
                    b_sub_garedient = -1 * label
                b = s_grad_update(b_sub_garedient, b, l_rate)
                w = s_grad_update(w_sub_garedient, w, l_rate)
        return w, b


def calculate_error(w, bias, x, y):
    right_classification_counter = 0
    for i in range(len(x)):
        predicted_label = (np.dot(x[i], w) + bias)
        if predicted_label * y[i] >= 0:
            right_classification_counter += 1
            # print("the error is: ",1 - (right_classification_counter / len(x)))
    return 1 - (right_classification_counter / len(x))


def create_f(a, b, c):
    def f_x(x):
        return a + b * x + c * x ** 2

    return f_x


def create_g(b, c):
    def grad_f(x):
        return 2 * c * x + b

    return grad_f


def main():
    # 100 linearly spaced numbers
    LinearSpace = np.linspace(-10, 10, 100)
    # the function, which is y = x^2 here
    f = create_f(1, 1, 1)
    # setting the axes at the centre
    fig = plt.figure()
    g_f = create_g(1, 1)
    # plot the function
    plt.plot(LinearSpace, f(LinearSpace), 'r')
    # show the plot
    # gradient_decent
    x, xt, fs = gradient_method(f, g_f, 10)
    # scatter
    plt.scatter(xt, fs, cmap='autumn')
    plt.show()
    # EX 3
    X, y = load_iris(return_X_y=True)
    X = X[y != 0]
    y = y[y != 0]
    y[y == 2] = -1
    X = X[:, 2:4]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
    lambdas = [0, 0.05, 0.1, 0.2, 0.5]
    errors_train = []
    errors_test = []
    margins = []
    for lam in lambdas:
        w, b = svm_with_sgd(X_train, y_train, lam, 1000, l_rate=0.01, sgd_type="practical")
        errors_train.append(calculate_error(w, b, X_train, y_train))
        errors_test.append(calculate_error(w, b, X_val, y_val))
        margins.append(1 / np.linalg.norm(w))
    fig, asx = plt.subplots()
    asx.set_xlabel('lamda')
    asx.set_ylabel('error rate')
    width = 0.3
    l1 = asx.bar(np.arange(len(lambdas)) - (width / 2), errors_train, width, color='blue', label="train error")
    l2 = asx.bar(np.arange(len(lambdas)) + (width / 2), errors_test, width, color='grey', label="test error")
    asx.set_xticks(np.arange(len(lambdas)))
    asx.set_xticklabels(lambdas)
    matplotlib.pyplot.legend(["train", "test"], bbox_to_anchor=(0.75, 1.15), ncol=2)
    plt.title("error comparison")
    plt.show()
    fig, asx = plt.subplots()
    asx.set_xlabel('lamda')
    asx.set_ylabel('margin')
    l1 = asx.bar(np.arange(len(lambdas)), margins, width, color='blue', label="margin")
    asx.set_xticks(np.arange(len(lambdas)))
    asx.set_xticklabels(lambdas)
    plt.title("margins comparison")
    plt.show()
    # e 2 z
    errors_train_p = []
    errors_test_p = []
    errors_train_t = []
    errors_test_t = []
    for epoc in range(10, 1010, 10):
        wp, bp = svm_with_sgd2(X_train, y_train, 0.05, epoc, 0.01, 'practical')
        wt, bt = svm_with_sgd2(X_train, y_train, 0.05, epoc, 0.01, 'theory')
        errors_train_p.append(calculate_error(wp, bp, X_train, y_train))
        errors_train_t.append(calculate_error(wt, bt, X_train, y_train))
        errors_test_t.append(calculate_error(wt, bt, X_val, y_val))
        errors_test_p.append(calculate_error(wp, bp, X_val, y_val))
    epochs = range(10, 1010, 10)
    p1 = plt.plot(epochs, errors_train_p, label="practical", color='blue')
    p2 = plt.plot(epochs, errors_train_t, label="theory", color='grey')
    plt.title("TRAIN: practical error vs theory error")
    matplotlib.pyplot.legend(["practical", "theory"], bbox_to_anchor=(0.75, 1.15), ncol=2)
    plt.show()

    p3 = plt.plot(epochs, errors_test_p, label="practical", color='blue')
    p3 = plt.plot(epochs, errors_test_t, label="theory", color='grey')
    plt.title("Test: practical error vs theory error")
    matplotlib.pyplot.legend(["practical", "theory"], bbox_to_anchor=(0.75, 1.15), ncol=2)
    plt.show()





if __name__ == "__main__":
    main()
