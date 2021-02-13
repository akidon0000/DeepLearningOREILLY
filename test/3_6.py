# coding: utf-8

from PIL import Image
from dataset1.mnist import load_mnist
import numpy as np
import os
import pickle
import sys
# sys.path.append("/Users/'akidon'/MacMini/DeepLearning/DeepLearning_OREILLY")
sys.path.append(os.pardir)  # パスに親ディレクトリ追加


def sigmoid(x):
    """シグモイド関数
    本の実装ではオーバーフローしてしまうため、以下のサイトを参考に修正。
    http://www.kamishima.net/mlmpyja/lr/sigmoid.html

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    # xをオーバーフローしない範囲に補正
    sigmoid_range = 34.538776394910684
    x2 = np.maximum(np.minimum(x, sigmoid_range), -sigmoid_range)

    # シグモイド関数
    return 1 / (1 + np.exp(-x2))


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


def load_test_data():
    """MNISTのテスト画像とテストラベル取得
        画像の値は0.0〜1.0に正規化済み。

    Returns:
        numpy.ndarray, numpy.ndarray: テスト画像, テストラベル
    """
    (x_train, t_train), (x_test, t_test) \
        = load_mnist(flatten=True, normalize=True)
    # print(str(x_train),":",str(t_train),":",str(x_test),":",str(t_test))
    print(x_train[0])
    return x_test, t_test


def load_sapmle_network():
    """サンプルの学習済み重みパラメーター取得

    Returns:
        dict: 重みとバイアスのパラメーター
    """
    with open("/Users/akidon/MacMini/Program-mini/DeepLearning/DeepLearning_OREILLY/test/sample_weight.pkl", "rb") as f:
    # with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    """ニューラルネットワークによる推論

    Args:
        network (dict): 重みとバイアスのパラメーター
        x (numpy.ndarray): ニューラルネットワークへの入力

    Returns:
        numpy.ndarray: ニューラルネットワークの出力
    """
    # パラメーター取り出し
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # ニューラルネットワークの計算（forward）
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def show_image(img):
    """イメージ表示

    Args:
        image (numpy.ndarray): 画像のビットマップ
    """
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# MNISTのテストデータの読み込み
x, t = load_test_data()

# サンプルの重みパラメーター読み込み
network = load_sapmle_network()

# 推論、認識精度算出
batch_size = 100 # バッチ処理の単位
accuracy_cnt = 0  # 正解数
error_image = None  # 認識できなかった画像
for i in range(0, len(x), batch_size):

    # バッチデータ準備
    x_batch = x[i:i + batch_size]

    # 推論
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)

    # 正解数カウント
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

    # 認識できなかった画像をerror_imageに連結
    for j in range(0, batch_size):
        if p[j] != t[i + j]:
            if error_image is None:
                error_image = x_batch[j]
            else:
                error_image = np.concatenate((error_image, x_batch[j]), axis=0)

print("認識精度:" + str(accuracy_cnt / len(x)))


# 認識できなかった画像を表示
error_image *= 255  # 画像の値は0.0〜1.0に正規化されているので、表示できるように0〜255に戻す。
show_image(error_image.reshape(28 * (len(x) - accuracy_cnt), 28))
