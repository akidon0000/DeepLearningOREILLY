# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    """入力が0を超えたら1を返すステップ関数

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    """シグモイド関数

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU関数

    Args:
        x (numpy.ndarray)): 入力

    Returns:
        numpy.ndarray: 出力
    """
    return np.maximum(0, x)


# 計算
x = np.arange(-5.0, 5.0, 0.01)  # stepはステップ関数が斜めに見えないように小さめ
y_step = step_function(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)

# グラフ描画
plt.plot(x, y_step, label="step")
plt.plot(x, y_sigmoid, linestyle="--", label="sigmoid")
plt.plot(x, y_relu, linestyle=":", label="ReLU")
plt.ylim(-0.1, 5.1)
plt.legend()
plt.show()
