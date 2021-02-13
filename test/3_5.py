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


def softmax(x):
    """ソフトマックス関数

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """

    # バッチ処理の場合xは(バッチの数, 10)の2次元配列になる。
    # この場合、ブロードキャストを使ってうまく画像ごとに計算する必要がある。
    if x.ndim == 2:

        # 画像ごと（axis=1）の最大値を算出し、ブロードキャストできるよにreshape
        c = np.max(x, axis=1).reshape(x.shape[0], 1)

        # オーバーフロー対策で最大値を引きつつ分子を計算
        exp_a = np.exp(x - c)

        # 分母も画像ごと（axis=1）に合計し、ブロードキャストできるよにreshape
        sum_exp_a = np.sum(exp_a, axis=1).reshape(x.shape[0], 1)

        # 画像ごとに算出
        y = exp_a / sum_exp_a

    else:

        # バッチ処理ではない場合は本の通りに実装
        c = np.max(x)
        exp_a = np.exp(x - c)  # オーバーフロー対策
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

    return y


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
