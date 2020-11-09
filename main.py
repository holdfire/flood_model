import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm, linear_model


model_names = ["kNN", "SVR", "LR", "RidgeR", "AdaBoost", "GBDT", "RF", "XGBoost"]

parser = argparse.ArgumentParser(description="flood inundation machine learning model. ")
parser.add_argument("--train_data", default="./examples/basin1/train.csv",
                    type=str, help="path to training data file. ")
parser.add_argument("--test_data", default="./examples/basin1/test.csv",
                    type=str, help="path to testing data file. ")
parser.add_argument( "-m", "--method", default="kNN",
                     type=str, help="selecting the machine learning method. ",
                     choices=model_names)
parser.add_argument('--seed', default=2020, type=int,
                    help='seed for initializing training. ')





def get_original_data(path):
    data = pd.read_csv(path)
    # 获取表头信息
    title = list(data.columns)
    data = data.astype("float32")
    data = np.array(data)
    # X为输入，即单轴压缩强度，弹性模量和泊松比， Y为多个输出，
    X = data[ :, :3]
    Y = data[ :, 3:8]
    return X, Y, title


def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)
    # 对输入数据X的每一列做归一化到[0, 1]区间
    X_train = (X_train - np.mean(X_train, axis=0, keepdims=True)) / (np.max(X_train, axis=0, keepdims=True) - np.min(X_train, axis=0, keepdims=True))
    X_test = (X_test - np.mean(X_train, axis=0, keepdims=True)) / (np.max(X_train, axis=0, keepdims=True) - np.min(X_train, axis=0, keepdims=True))
    return X_train, X_test, y_train, y_test


def regression(X_train, y_train, regressor = "SVR"):
    # 参考https://scikit-learn.org/stable/modules/svm.html#svm-regression
    if regressor == "SVR":
        reg = svm.SVR(kernel='rbf', gamma="scale")
    # 参考：https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
    elif regressor == "kNN":
        reg = KNeighborsRegressor(n_neighbors=4, weights="uniform")
    elif regressor == "LinearRegression":
        reg = linear_model.LinearRegression()
    elif regressor == "Lasso":
        reg = linear_model.Lasso(alpha=0.1)
    elif regressor == "Ridge":
        reg = linear_model.Ridge(alpha=0.5)
    reg.fit(X_train, y_train)
    return reg

def model_evaluation(reg, X_test, y_test):
    y_predict = reg.predict(X_test)
    score = r2_score(y_test, y_predict)
    return score, y_predict




if __name__ == "__main__":
    # 本程序唯一的参数，可选择SVR, kNN, LinearRegression, Lasso, Ridge
    regressor = "SVR"

    # 注意：将数据和本代码放在同一个目录下
    path_train = "data_train.csv"
    # X是shape为（32，3）的输入，Y为（32，5）的输出
    X, Y, title = get_original_data(path_train)
    print("输入数据的shape为: ", X.shape, "输出数据的shape为：", Y.shape)

    # 存储测试集的结果
    result_test = []

    # 存储预测集的结果
    path_predict = "data_predict.csv"
    X_new = pd.read_csv(path_predict)
    X_new = X_new.astype("float32")
    X_new = np.array(X_new)
    result_predict = X_new
    print(X_new)


    # 开始训练模型
    for i in range(Y.shape[1]):
        y = Y[:, i]
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        # 选择回归模型，默认用支持向量回归
        reg = regression(X_train, y_train, regressor)

        # 测试集结果
        score, y_predict = model_evaluation(reg, X_test, y_test)
        print("当前输出为：", title[X.shape[1] + i], "  选用的回归模型为：", regressor, "测试集得分为：", score)
        result_test.append(y_test)
        result_test.append(y_predict)

        # 预测结果
        cur_model = []
        for i in range(X_new.shape[0]):
            x_new = X_new[i].reshape((1,-1))
            print(x_new)
            y_new = reg.predict(x_new)
            cur_model.append(y_new)
        cur_model = np.array(cur_model)
        result_predict = np.hstack((result_predict, cur_model))


    # 保存测试集结果
    result_test = np.array(result_test).transpose()
    np.savetxt("./result_test.csv", result_test)

    # 保存预测集结果
    print(result_predict)
    np.savetxt("./result_predict.csv", result_predict)




