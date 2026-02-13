import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
import random

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback, EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def create_layers(num_layers: int):
    layers = []
    params = [[[] for i in range(3)] for i in range(num_layers)]

    for layer in range(num_layers):
        layer_type = st.selectbox(f"**{layer+2}層目**", ["全結合層", "畳み込み層", "平坦化", "プーリング層", "ドロップアウト層"])
        param = params[layer]
        if layer_type == "畳み込み層":
            # 畳み込み層のパラメータ設定
            col1, col2, col3 = st.columns(3)
            param[0] = col1.selectbox("*Filters(フィルタの数)*", [16, 32, 64, 128, 256, 512], key="filter_"+str(layer))
            param[1] = col2.number_input("*Kernel Size(フィルタの寸法)(N x N)*", min_value=1, max_value=5, key="kernel_"+str(layer))
            param[2] =col3.selectbox("*活性化関数*", ["relu", "softmax"], key="act_cnn_"+str(layer))
            layers.append([layer_type, layer])
        elif layer_type == "全結合層":
            # 全結合層のパラメータ設定
            col1, col2, col3 = st.columns(3)
            param[0] = col1.number_input("*ユニット数*", min_value=2, max_value=512, key="units_"+str(layer))
            param[1] = col2.selectbox("*活性化関数*", ["relu", "softmax"], key="act_dense_"+str(layer))
            param[2] = None
            layers.append([layer_type, layer])
        elif layer_type == "プーリング層":
            # プーリング層のパラメータ設定
            col1, col2, col3 = st.columns(3)
            param[0] = col1.number_input("*プーリングのサイズ(N x N)*", min_value=1, max_value=10, key="pooling_"+str(layer))
            param[1] = None
            param[2] = None
            layers.append([layer_type, layer])
        elif layer_type == "ドロップアウト層":
            # ドロップアウト層のパラメータ設定
            param[0] = st.slider("ドロップアウト率(出力を減らす)", min_value=0.00, max_value=1.00, step=.01, key="dropout_"+str(layer))
            layers.append([layer_type, layer])
        else:
            layers.append([layer_type, layer])

    can_create = False
    last_activation = params[-1][:2]

    # モデル作成ができるかの判定(中間層は見ない)
    if last_activation[0] == 10 and last_activation[1] == "softmax":
        can_create = True
    else:
        st.warning("最後のレイヤーは全結合層, 活性化関数はSoftmax, ユニット数は10にしてください")
        can_create = False

    return layers, params, can_create

# モデルを作成する処理
def create_model(X_train, y_train, X_val, log_area):
    # ログを入れる変数
    metrics_data = []

    # 訓練データをさらに分割
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=42)

    model = Sequential()

    for layer, param in zip(layers, params):
        if layer[0] == "畳み込み層" and layer[1] == 0:
            # 画像データの形を整えて正規化
            X_train, X_test = X_train.reshape(-1, 28, 28, 1) / 255.0, X_test.reshape(-1, 28, 28, 1) / 255.0
            model.add(Conv2D(
                filters=param[0],
                kernel_size=(param[1], param[1]),
                input_shape=(28, 28, 1),
                padding="same",
                activation=param[2])
            )
        elif layer[0] == "畳み込み層":
            model.add(Conv2D(
                filters=param[0],
                kernel_size=(param[1], param[1]),
                activation=param[2]
                )
            )
        elif layer[0] == "全結合層" and layer[1] < 2:
            # 画像データの形を整えて正規化
            X_train, X_test = X_train.reshape(-1, 28, 28, 1) / 255.0, X_test.reshape(-1, 28, 28, 1) / 255.0
            model.add(Dense(
                units=param[0],
                input_shape=(28, 28, 1),
                activation=param[1]
                )
            )
        elif layer[0] == "全結合層":
            model.add(Dense(
                units=param[0],
                activation=param[1]
                )
            )
        elif layer[0] == "プーリング層":
            model.add(MaxPooling2D(pool_size=param[0]))
        elif layer[0] == "平坦化":
            model.add(Flatten())

    # モデルの最適化アルゴリズム, 損失関数, 評価関数を設定
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 学習過程をリアルタイムに表示するcallbackの設定
    stream_it_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: (
            metrics_data.append({
                "epoch": epoch+1, 
                "acc": logs["accuracy"], 
                "loss": logs["loss"], 
                "val_acc": logs["val_accuracy"],
                "val_loss": logs["val_loss"]
            }),
            log_area.code(pd.DataFrame(metrics_data).tail(10).to_string(col_space=10, index=False))
        )
    )
    callbacks = list(st.session_state["callbacks"].values())
    callbacks.append(stream_it_callback)

    # 学習の実行
    data_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=.2,
        height_shift_range=.2,
        zoom_range=.2,
        horizontal_flip=False,
        vertical_flip=False
    )
    data_gen.fit(X_train)

    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=callbacks
    )

    return model, history, X_val

# 精度のプロット処理
def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc)+1)

    st.subheader("訓練データと検証用データでの精度の比較", width="stretch")
    
    # accuracyの比較
    compare_acc = plt.figure()
    plt.title("Comparison of Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(epochs, acc, "-r")
    plt.plot(epochs, val_acc, "-b")
    plt.grid(True)

    # lossの比較
    compare_loss = plt.figure()
    plt.title("Comparison of Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(epochs, loss, "-r")
    plt.plot(epochs, val_loss, "-b")
    plt.grid(True)

    col1, col2 = st.columns(2)
    col1.pyplot(compare_acc)
    col2.pyplot(compare_loss)

# モデルfit時のcallback設定
@st.dialog("高度な学習設定")
def fit_option():
    callbacks = {}
    csvlogger_value = False
    earlystopping_value = False
    modelcheckpoint_value = False

    # 最新のcallback設定のtypeの取得
    type_object = st.session_state["callbacks"].keys()
    for class_object in st.session_state["callbacks"].values():
        # 既にcallbackにチェックを入れていたら初期状態をチェック済みにする処理
        if type(class_object) in type_object:
            if "CSVLogger" in str(type(class_object)):
                csvlogger_value = True
            elif "EarlyStopping" in str(type(class_object)):
                earlystopping_value = True
            elif "ModelCheckpoint" in str(type(class_object)):
                modelcheckpoint_value = True

    # 送信部分
    with st.form("fit_setting", clear_on_submit=True):
        csvlogger = st.checkbox("学習の経過をcsv保存", value=csvlogger_value)
        earlystopping = st.checkbox("学習のストップ", value=earlystopping_value)
        modelcheckpoint = st.checkbox("ベストモデルの保存", value=modelcheckpoint_value)

        submitted = st.form_submit_button("保存")
        # "保存"ボタンが押されたら
        if submitted:
            if csvlogger:
                # 学習過程をcsv形式で保存
                callback = CSVLogger("./logs/training.csv")
                callbacks[type(callback)] = callback
            if earlystopping:
                # fit中に学習が進まなくなったら学習を止める            
                callback = EarlyStopping(monitor="val_loss", patience=5, verbose=0, mode="auto")
                callbacks[type(callback)] = callback
            if modelcheckpoint:
                callback = ModelCheckpoint(
                    filepath="./models/best_model.keras",
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    mode="min",
                    verbose=1
                )
                callbacks[type(callback)] = callback
            st.session_state["callbacks"] = callbacks
            st.success("保存しました")
            time.sleep(.5)
            st.rerun()

# 学習に用いる画像の表示・確認
@st.dialog("画像確認用")
def plot_show_image():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train)
    data_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=.2,
        height_shift_range=.2,
        zoom_range=.2,
        horizontal_flip=False,
        vertical_flip=False
    )
    data_gen.fit(X_train)
    x_batch, y_batch = next(data_gen.flow(X_train, y_train, batch_size=32))

    # 最初の10枚を表示
    fig, ax = plt.subplots()
    index = random.randint(0, 31)
    # (28, 28, 1) -> (28, 28) に戻して表示
    plt.imshow(x_batch[index].reshape(28, 28), cmap='gray')
    fig.suptitle(f"MNIST Data : {np.argmax(y_batch[index])}")
    st.pyplot(fig)

# 用語の説明
@st.dialog("用語の説明")
def explain_words():
    st.markdown(
r"""
## レイヤー名について
- **全結合層(Dense)**:<br>
　全結合層は以下の画像のように一つのユニットが次の層のすべてのユニットと繋がっている層である. 入力は一次元のデータになるため, 平坦化が必要.<br>
　画像認識では最終段階での分類を行う上で非常に効果的に作用する.
![Dense](https://thumb.ac-illust.com/f9/f979ebc8d06f617f98a4640b9424e49c_t.jpeg)

- **畳み込み層(Convolution)**:<br>
　畳み込み層とは二次元データに対してカーネル(フィルタ)を用いて行列計算を行い, データの特徴を捉えることのできる手法である. 引数にはカーネルの数やサイズ, ストライドなどがあり適切な値だととても高い精度が期待できる.<br>
　カーネルは抽出する特徴(縦線, 横線, 斜線など)に反応し, 層を重ねることで低次の特徴から高次の特徴(顔や物体など)を捉えることができる.
![Conv](https://farml1.com/wp-content/uploads/2021/05/image-21.png)
>画像出典: 【CNN】畳み込みニューラルネットワークを理解する①(https://farml1.com/cnn_1/)

- **プーリング層(Pooling)**:<br>
　プーリング層はダウンサンプリングとも呼ばれ, 畳み込み層の後に使用される層である. この層は次元数の削減を行い, パラメータの量を減らす. そのため多くの情報が失われるが, 複雑さが低減し過学習を防ぐことにつながる.<br>
　プーリングの種類には複数あり, 範囲内の最大値を出力とするものや平均を出力とするものなどがある.
　また, 対象がどこにいても判別できるようになる(移動不変性).
![Pooling](https://resources.zero2one.jp/images/questions/ai_quiz_v2_00311.jpeg)
>画像出典: プーリング(https://zero2one.jp/ai-word/pooling/)

- **平坦化(Flatten)**:<br>
　平坦化は二次元以上のデータ(画像データなど)を一次元に変換することである. 全結合層の入力は一次元データのため, 全結合層を使用する前に平坦化を行う.

- **ドロップアウト(Dropout)**:<br>
　ドロップアウトは学習時の過学習を防止する手法である. 主な仕組みは, 事前に設定した割合で入力ノードの一つの出力が0(無効)になる.
![Dropout](https://miro.medium.com/v2/resize:fit:720/format:webp/1*iWQzxhVlvadk6VAJjsgXgg.png)
>画像出典: Dropout in (Deep) Machine learning(https://zero2one.jp/ai-word/pooling/)


## 活性化関数について
- **シグモイド関数**:<br>
　シグモイド関数は(0, 0.5)の位置を中心とした点対称なS字型の曲線で, 値は0~1の間を取る. ディープラーニングで用いられるシグモイド関数は標準シグモイド関数と呼ばれ, 以下のように表される.
    <math xmlns="http://www.w3.org/1998/Math/MathML" data-latex="f(x)&#xA0;=&#xA0;\frac{1}{1+e^{-x}}" display="block">
    <mrow>
        <mi data-latex="f">f</mi>
        <mo>&#x2061;</mo>
        <mrow>
        <mo data-latex="(" stretchy="false">(</mo>
        <mi data-latex="x">x</mi>
        <mo data-latex=")" stretchy="false">)</mo>
        </mrow>
    </mrow>
    <mo data-latex="=">=</mo>
    <mfrac data-latex="\frac{1}{1+e^{-x}}">
        <mn data-latex="1">1</mn>
        <mrow data-latex="1+e^{-x}">
        <mn data-latex="1">1</mn>
        <mo data-latex="+">+</mo>
        <msup data-latex="e^{-x}">
            <mi data-latex="e">e</mi>
            <mrow data-mjx-texclass="ORD" data-latex="{-x}">
            <mo data-latex="-">&#x2212;</mo>
            <mi data-latex="x">x</mi>
            </mrow>
        </msup>
        </mrow>
    </mfrac>
    </math>
　ニューラルネットワークではパラメータの更新に誤差逆伝播法(バックプロパゲーション)を利用するが, この際に簡単に微分を計算できるため, 活性化関数によく用いられる. しかし, 微分を続けると微分後の値が0に近づいてしまう弱点がある(**勾配消失**).
![Sigmoid](https://resources.zero2one.jp/2023/03/d099d886ed65ef765625779e628d2c5f.png.webp)

- **ReLU関数**:<br>
　ReLU関数は人工知能の学習でよく使用される関数で, 以下のように定義される.
    <math xmlns="http://www.w3.org/1998/Math/MathML" data-latex="f(x)= \begin{cases}         0    \quad     (x&lt;0&#x306E;&#x3068;&#x304D;)             \\[0.5em]          x    \quad    (x\geqq0&#x306E;&#x3068;&#x304D;)         \\  \end{cases} " display="block">
        <mrow>
            <mi data-latex="f">f</mi>
            <mo>&#x2061;</mo>
            <mrow>
            <mo data-latex="(" stretchy="false">(</mo>
            <mi data-latex="x">x</mi>
            <mo data-latex=")" stretchy="false">)</mo>
            </mrow>
        </mrow>
        <mo data-latex="=">=</mo>
        <mrow data-mjx-texclass="INNER" data-latex-item="{cases}" data-latex="{cases}">
            <mrow>
            <mo data-mjx-texclass="OPEN">{</mo>
            <mtable columnspacing="1em" rowspacing="0.7em 0.2em" columnalign="left left">
                <mtr>
                <mtd>
                    <mrow>
                    <mn data-latex="0">0</mn>
                    <mstyle scriptlevel="0" data-latex="\quad">
                        <mspace width="1em"></mspace>
                    </mstyle>
                    <mrow>
                        <mo data-latex="(" stretchy="false">(</mo>
                        <mrow>
                        <mi data-latex="x">x</mi>
                        <mo data-latex="&lt;">&lt;</mo>
                        <mrow>
                            <mn data-latex="0">0</mn>
                            <mo>&#x2062;</mo>
                            <mi mathvariant="normal" data-latex="&#x306E;">&#x306E;</mi>
                            <mo>&#x2062;</mo>
                            <mi mathvariant="normal" data-latex="&#x3068;">&#x3068;</mi>
                            <mo>&#x2062;</mo>
                            <mi mathvariant="normal" data-latex="&#x304D;">&#x304D;</mi>
                        </mrow>
                        </mrow>
                        <mo data-latex=")" stretchy="false">)</mo>
                    </mrow>
                    </mrow>
                </mtd>
                </mtr>
                <mtr>
                <mtd>
                    <mrow>
                    <mi data-latex="x">x</mi>
                    <mstyle scriptlevel="0" data-latex="\quad">
                        <mspace width="1em"></mspace>
                    </mstyle>
                    <mrow>
                        <mo data-latex="(" stretchy="false">(</mo>
                        <mrow>
                        <mi data-latex="x">x</mi>
                        <mo data-latex="\geqq">&#x2267;</mo>
                        <mrow>
                            <mn data-latex="0">0</mn>
                            <mo>&#x2062;</mo>
                            <mi mathvariant="normal" data-latex="&#x306E;">&#x306E;</mi>
                            <mo>&#x2062;</mo>
                            <mi mathvariant="normal" data-latex="&#x3068;">&#x3068;</mi>
                            <mo>&#x2062;</mo>
                            <mi mathvariant="normal" data-latex="&#x304D;">&#x304D;</mi>
                        </mrow>
                        </mrow>
                        <mo data-latex=")" stretchy="false">)</mo>
                    </mrow>
                    </mrow>
                </mtd>
                </mtr>
            </mtable>
            </mrow>
            <mo data-mjx-texclass="CLOSE" fence="true" stretchy="true" symmetric="true"></mo>
        </mrow>
    </math>
　この関数もシグモイド関数同様計算が容易で, 活性化関数としてよく用いられる関数の一つである. この関数の特徴は, シグモイド関数で問題だった勾配消失が起こりにくいことである.
![ReLU](https://resources.zero2one.jp/2023/03/6f8259a3a4c3567d41e4f709cef288ed.png.webp)

- **tanh関数**:<br>
　tanh関数(ハイパボリックタンジェント関数)も活性化関数によく用いられる関数で, 形状はシグモイド関数をとても似ている. しかし, 0～1までの値をとるシグモイド関数とは異なり出力は-1～1までの値をとる. そのためシグモイド関数よりも勾配消失が起こりにくいメリットがある. しかし, この関数も勾配消失の問題がある.
![tanh](https://resources.zero2one.jp/2023/03/08810331fa90a2994aadd459a0cbe688.png.webp)

- **ソフトマックス関数**:<br>
　ソフトマックス関数は, 出力の合計が1になるように入力値を変換する関数である. 合計が1になることから, 得られた出力はそれぞれの確立としてみることができ, 分類問題を解くための最終層に用いられる.<br>
　数式は以下のようになる.
    <math xmlns="http://www.w3.org/1998/Math/MathML" data-latex="y_i&#xA0;=&#xA0;\frac{exp(x_i)}{\sum_{k=1}^{n}exp(x_i)}" display="block">
    <msub data-latex="y_i">
        <mi data-latex="y">y</mi>
        <mi data-latex="i">i</mi>
    </msub>
    <mo data-latex="=">=</mo>
    <mfrac data-latex="\frac{exp(x_i)}{\sum_{k=1}^{n}exp(x_i)}">
        <mrow data-latex="exp(x_i)">
        <mi data-latex="e">e</mi>
        <mo>&#x2062;</mo>
        <mi data-latex="x">x</mi>
        <mo>&#x2062;</mo>
        <mrow>
            <mi data-latex="p">p</mi>
            <mo>&#x2061;</mo>
            <mrow>
            <mo data-latex="(" stretchy="false">(</mo>
            <msub data-latex="x_i">
                <mi data-latex="x">x</mi>
                <mi data-latex="i">i</mi>
            </msub>
            <mo data-latex=")" stretchy="false">)</mo>
            </mrow>
        </mrow>
        </mrow>
        <mrow data-latex="\sum_{k=1}^{n}exp(x_i)">
        <munderover data-latex="\sum_{k=1}^{n}">
            <mo data-latex="\sum">&#x2211;</mo>
            <mrow data-mjx-texclass="ORD" data-latex="{k=1}">
            <mi data-latex="k">k</mi>
            <mo data-latex="=">=</mo>
            <mn data-latex="1">1</mn>
            </mrow>
            <mrow data-mjx-texclass="ORD" data-latex="{n}">
            <mi data-latex="n">n</mi>
            </mrow>
        </munderover>
        <mrow>
            <mi data-latex="e">e</mi>
            <mo>&#x2062;</mo>
            <mi data-latex="x">x</mi>
            <mo>&#x2062;</mo>
            <mrow>
            <mi data-latex="p">p</mi>
            <mo>&#x2061;</mo>
            <mrow>
                <mo data-latex="(" stretchy="false">(</mo>
                <msub data-latex="x_i">
                <mi data-latex="x">x</mi>
                <mi data-latex="i">i</mi>
                </msub>
                <mo data-latex=")" stretchy="false">)</mo>
            </mrow>
            </mrow>
        </mrow>
        </mrow>
    </mfrac>
    </math>

- そのほかにも, leaky ReLU関数やGELU関数といった関数も活性化関数としてよく用いられる.

>参照: 深層学習において重要なシグモイドやReLUなどの活性化関数の特徴を解説(https://zero2one.jp/learningblog/deep-learning-activation-function/)

## ネットワークの構造について
- **DNN(*Deep Neural Network*)**:<br>
　DNNは人間の脳の神経回路を模倣したモデルで, 入力層, 中間層(隠れ層), 出力層からなる. 隠れ層が多層になることでモデルがより複雑になり, より複雑なデータを処理できるようになる.<br>
　入力は主に一次元のベクトルデータを扱う. DNNでも画像認識は可能だが, 画像データをそのまま一次元データにしているため二次元的な情報(右上に模様がある)が失われてしまったり, 対象の位置がずれてしまうと判別しにくくなるといったデメリットがある.
![Dense](https://thumb.ac-illust.com/f9/f979ebc8d06f617f98a4640b9424e49c_t.jpeg)

- **CNN(*Convolutional Neural Network*)**:<br>
　CNNは画像認識などの空間データに特化したモデルで, DNNとは違い, 元のデータ構造を保持したまま処理できる. CNNの構造は中間層に「畳み込み層」と「プーリング層」を用いて構成される.<br>
　CNNでは畳み込み層やプーリング層を用いることでDNNでのデメリットを無くし, より高い精度での画像認識を行えることができる.
![Dropout](https://deepage.net/img/convolutional_neural_network/cnn_thumb.jpg)
>画像出典: 定番のConvolutional Neural Networkをゼロから理解する(https://deepage.net/deep_learning/2016/11/07/convolutional_neural_network.html)

""", unsafe_allow_html=True)

# -- UI --
st.title("モデルの保存")
st.write("MNISTを使用したモデルの作成を行います.")
st.write("**1層目** : *Input(28×28×1)*")

num_layers = st.slider("層の数", min_value=3, max_value=30)
layers, params, can_create = create_layers(num_layers)
is_create_model = st.button("開始", disabled=can_create is not True)

with st.sidebar:
    epochs = st.slider("エポック数", min_value=1, max_value=100, value=10, disabled=is_create_model)
    batch_size = st.selectbox("バッチサイズ", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], index=6, disabled=is_create_model)

    st.space("medium")
    if st.button("高度な学習設定", width="stretch", disabled=is_create_model):
        fit_option()
    if st.button("学習画像の確認", width="stretch", disabled=is_create_model):
        plot_show_image()
    if st.button("用語の説明", width="stretch", disabled=is_create_model):
        explain_words()

if is_create_model:
    try:
        with st.status("モデルの作成中", expanded=True):
            log_area = st.empty()

            # csvファイルがあったら削除
            if os.path.isfile("./logs/training.csv"):
                os.remove("./logs/training.csv")

            # データの取得, 分割
            mnist = tf.keras.datasets.mnist
            (X_train, y_train), (X_val, y_val) = mnist.load_data()

            # ラベルを"1", "2"などから[0, 1, 0, 0,..]のようなone-hot表現に変換
            y_train, y_val = to_categorical(y_train), to_categorical(y_val)

            #モデルを作成
            model, history, X_val = create_model(X_train, y_train, X_val, log_area)

            # 未知データでの評価
            score = model.evaluate(X_val, y_val, verbose=0)
            log_area.empty()

            plot_history(history)
            
            st.subheader("テスト用データでの精度")
            col1, col2 = st.columns(2)
            col1.write(f"**Accuracy**: {score[1]}")
            col2.write(f"**Loss**: {score[0]}")
            
            # モデルを一時保存
            model.save("./models/my_model.keras")

        with st.status("モデルの作成完了", state="complete"):
            col1, col2, col3 = st.columns(3)
            # モデルのダウンロード処理
            with open("./models/my_model.keras", "rb") as f:
                    col1.download_button("モデルのダウンロード",
                                    data=f,
                                    file_name="model.keras",
                                    icon=":material/download:"
                                )
            # ベストモデルの保存
            callbacks = st.session_state["callbacks"].keys()
            if "ModelCheckpoint" in str(callbacks):
                # モデルのダウンロード処理
                with open("./models/best_model.keras", "rb") as f:
                    col2.download_button("ベストモデルのダウンロード",
                                    data=f,
                                    file_name="best_model.keras",
                                    icon=":material/download:"
                                )

    except Exception as e:
        st.error("モデル作成でエラーが発生しました.\nパラメータの確認をしてください")
        st.error(e)

    # csvファイルがあったら表示
    if os.path.isfile("./logs/training.csv"):
        with open("./logs/training.csv", "rb") as f:
            df = pd.read_csv(f)
            st.subheader("学習過程")
            st.dataframe(df, hide_index=True)
    else:
        st.warning("学習の詳細をCSVでダウンロードするには学習設定の変更をしてください")

    st.button("やり直す")


