import numpy as np
from data import *


class DecisionTree:
    def __init__(self,trainx, trainy):
        self.train_x = trainx
        self.train_y = trainy
        self.t = self.tree(self.train_x, self.train_y)

    def mostlabel(self, labels):
        labeldict = {}
        for label in labels:
            if label in labels:
                labeldict[label] += 1
            else:
                labeldict[label] = 1
        labelsort = sorted(labels.items(), key=lambda x: x[1], reverse=True)
        return labelsort[0][0]

    def h_d(self, labels):
        h_d = 0
        labelsset = set(labels)
        for label in labelsset:
            p = labels[labels == label].size / labels.size
            h_d += -p * np.log(p)
        return h_d

    def h_d_a(self, datas, labels):
        h_d_a = 0
        datasset = set(list(datas))
        for data in datasset:
            h_d_a += datas[datas == data].size / datas.size * self.h_d(labels[datas == data])
        return h_d_a

    def bestfeature(self, datas, labels):
        feature_num = datas.shape[1]
        feature_index = -1
        g_d_a = -float("inf")
        for i in range(feature_num):
            h_d = self.h_d(labels)
            data_feature = datas[:, i]
            g = h_d - self.h_d_a(data_feature, labels)
            if g_d_a < g:
                g_d_a = g
                feature_index = i
        return g_d_a, feature_index

    def get_sub_data(self, datas, labels, feature_index, feature_value):
        sub_data = []
        sub_label = []
        for i in range(datas.shape[0]):
            if datas[i][feature_index] == feature_value:
                sub_data.append(np.concatenate((datas[i][:feature_index], datas[i][feature_index+1:])))
                sub_label.append(labels[i])
        sub_data = np.array(sub_data)
        sub_label = np.array(sub_label)
        return sub_data, sub_label

    def tree(self, datas, labels):
        epsilon = 1e-2
        labelsset = set(labels)
        if len(labelsset) == 1:
            return labels[0]
        if datas.shape[1] == 1:
            return self.mostlabel(labels)
        _, feature_index = self.bestfeature(datas, labels)
        if _ < epsilon:
            return self.mostlabel(labels)
        tree_dict = {feature_index: {}}
        print(datas[:, feature_index].shape)
        features = set(list(datas[:, feature_index]))
        for f in features:
            sub_data = self.get_sub_data(datas, labels, feature_index, f)[0]
            sub_label = self.get_sub_data(datas, labels, feature_index, f)[1]
            tree_dict[feature_index][f] = self.tree(sub_data, sub_label)
        return tree_dict


def predict(data, model):
    tree = model.t
    f_value = 0
    while True:
        if type(tree) == type({}):
            if len(tree) == 1:
                (key, value), = tree.items()
                if type(value) == type({}):
                    tree = value
                    f_value = data[key]
                    data = np.concatenate((data[:key], data[key + 1:]))
                else:
                    return value
            else:
                for k, v in tree.items():
                    if k == f_value:
                        if type(v) == type({}):
                            tree = v
                        else:
                            return v
        else:
            return tree


if __name__ == "__main__":
    data = get_cifar_data(num_training=10,num_test=10)
    train_x = data["x_train"].reshape(10,-1)
    train_y = data["y_train"]
    test_x = data["x_test"].reshape(10,-1)
    test_y = data["y_test"]
    tree = DecisionTree(train_x, train_y)
    r_num = 0
    for i in range(test_y.shape[0]):
        result = predict(test_x[i], tree)
        if result == test_y[i]:
            r_num += 1
    acc = r_num / test_y.shape[0]
    predict(acc)