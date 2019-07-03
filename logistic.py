import numpy as np
from data import *

# 多分类逻辑回归
class L_R():
    def __init__(self, shape, n_classes=10):
        self.classes = n_classes
        n_sample, input_dim = shape
        self.params = dict()
        self.params["w"] = 1e-3 * np.random.randn(input_dim, n_classes).astype("float32")
        self.params["b"] = 1e-3 * np.random.randn(n_classes).astype("float32")

    def loss(self, x, y=None):
        z = np.dot(x, self.params["w"]) + self.params["b"]
        # print(z)
        y_hat = z - np.max(z, axis=1, keepdims=True)
        y_softmax_sum = np.sum(np.exp(y_hat), axis=1, keepdims=True)
        y_probs = np.exp(y_hat) / y_softmax_sum
        if y is None:
            return y_probs
        n = y_hat.shape[0]
        # print(y_probs)
        loss = -np.sum(np.log(y_probs[range(n), y] + 1e-5))/n
        dz = y_probs.copy()
        dz[range(n), y] -= 1
        dx = np.dot(dz, self.params["w"].T)
        dw = np.dot(x.T, dz)
        db = np.sum(dz, axis=0)
        grad = {
               "w": dw, "b": db
        }
        return loss, grad


def train(x, y, net, epoch=1, learning_rate=1e-3):
    model = net
    for i in range(epoch):
        mask = np.random.choice(x.shape[0], 128)
        batch_x = x[mask]
        batch_y = y[mask]
        loss, grad = model.loss(batch_x, batch_y)
        for k, v in grad.items():
            model.params[k] -= learning_rate * v
        if i % 100 == 0:
            print(loss)


if __name__ == "__main__":
    dataset = get_cifar_data(num_training=40000, num_test=100)
    x_train = dataset["x_train"].reshape(40000, -1)
    y_train = dataset["y_train"]
    x_test = dataset["x_test"].reshape(100, -1)
    y_test = dataset["y_test"]
    print(x_train.shape)
    model = L_R(x_train.shape, 10)
    train(x_train, y_train, model, 50000)
    y_pred = np.argmax(model.loss(x_test), axis=1)
    y_accurate = np.mean(y_pred == y_test)
    print(y_accurate)