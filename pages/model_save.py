import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split


def create_layers(num_layers: int):
    layers = []
    params = [[[] for i in range(3)] for i in range(num_layers)]

    for layer in range(num_layers):
        layer_type = st.selectbox(f"**{layer+1}層目**", ["Dense", "Conv2D", "Flatten", "MaxPooling2D", "Dropout"])
        param = params[layer]
        if layer_type == "Conv2D":
            # 畳み込み層のパラメータ設定
            col1, col2, col3 = st.columns(3)
            param[0] = col1.selectbox("*Filters(フィルタの数)*", [16, 32, 64, 128, 256, 512], key="filter_"+str(layer))
            param[1] = col2.number_input("*Kernel Size(フィルタの寸法)(N x N)*", min_value=1, max_value=5, key="kernel_"+str(layer))
            param[2] =col3.selectbox("*活性化関数*", ["relu", "softmax"], key="act_cnn_"+str(layer))
            layers.append([layer_type, layer])
        elif layer_type == "Dense":
            # 全結合層のパラメータ設定
            col1, col2, col3 = st.columns(3)
            param[0] = col1.number_input("*ユニット数*", min_value=2, max_value=512, key="units_"+str(layer))
            param[1] = col2.selectbox("*活性化関数*", ["relu", "softmax"], key="act_dense_"+str(layer))
            param[2] = None
            layers.append([layer_type, layer])
        elif layer_type == "MaxPooling2D":
            # プーリング層のパラメータ設定
            col1, col2, col3 = st.columns(3)
            param[0] = col1.number_input("*プーリングのサイズ(N x N)*", min_value=1, max_value=10, key="pooling_"+str(layer))
            param[1] = None
            param[2] = None
            layers.append([layer_type, layer])
        elif layer_type == "Flatten":
            layers.append([layer_type, layer])
        elif layer_type == "Dropout":
            # ドロップアウト層のパラメータ設定
            param[0] = st.slider("ドロップアウト率(出力を減らす)", min_value=0.00, max_value=1.00, step=.01, key="dropout_"+str(layer))
            layers.append([layer_type, layer])

    can_create = False
    last_activation = params[-1][:2]
    first_activation = layers[0][0]

    # モデル作成ができるかの判定(中間層は見ない)
    if last_activation[0] == 10 and last_activation[1] == "softmax" and (first_activation == "Dense" or first_activation == "Conv2D"):
        can_create = True
    elif last_activation[0] == 10 and last_activation[1] == "softmax":
        st.warning("最初のレイヤーはDenseかConv2Dにしてください")
        can_create = False
    elif first_activation == "Dense" or first_activation == "Conv2D":
        st.warning("最後のレイヤーはDense, 活性化関数はSoftmax, ユニット数は10にしてください")
        can_create = False
    else:
        st.warning("最初のレイヤーはDenseかConv2Dにしてください")
        st.warning("最後のレイヤーはDense, 活性化関数はSoftmax, ユニット数は10にしてください")
        can_create = False
    return layers, params, can_create

# モデルを作成する処理
def create_model(X_train, y_train, X_val, log_area):
    # ログを入れる変数
    metrics_data = []

    # 訓練データをさらに分割
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=42, stratify=y_train)

    model = Sequential()

    for layer, param in zip(layers, params):
        if layer[0] == "Conv2D" and layer[1] == 0:
            # 画像データの形を整えて正規化
            X_train, X_test = X_train.reshape(-1, 28, 28, 1) / 255.0, X_test.reshape(-1, 28, 28, 1) / 255.0
            model.add(Conv2D(
                filters=param[0],
                kernel_size=(param[1], param[1]),
                input_shape=(28, 28, 1),
                padding="same",
                activation=param[2])
            )
        elif layer[0] == "Conv2D":
            model.add(Conv2D(
                filters=param[0],
                kernel_size=(param[1], param[1]),
                activation=param[2]
                )
            )
        elif layer[0] == "Dense" and layer[1] == 0:
            # 画像データの形を整えて正規化 
            X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0
            X_val = X_val.reshape(-1, 784)
            model.add(Dense(
                units=param[0],
                input_shape=(784, ),
                activation=param[1]
                )
            )
        elif layer[0] == "Dense":
            model.add(Dense(
                units=param[0],
                activation=param[1]
                )
            )
        elif layer[0] == "MaxPooling2D":
            model.add(MaxPooling2D(pool_size=param[0]))
        elif layer[0] == "Flatten":
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
    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(X_test, y_test),
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
@st.dialog("学習設定")
def fit_option():
    callbacks = {}
    csvlogger_value = False
    earlystopping_value = False

    # 最新のcallback設定のtypeの取得
    type_object = st.session_state["callbacks"].keys()
    for class_object in st.session_state["callbacks"].values():
        # 既にcallbackにチェックを入れていたら初期状態をチェック済みにする処理
        if type(class_object) in type_object:
            if "CSVLogger" in str(type(class_object)):
                csvlogger_value = True
            elif "EarlyStopping" in str(type(class_object)):
                earlystopping_value = True

    # 送信部分
    with st.form("fit_setting", clear_on_submit=True):
        csvlogger = st.checkbox("学習の経過をcsv保存", value=csvlogger_value)
        earlystopping = st.checkbox("学習のストップ", value=earlystopping_value)

        submitted = st.form_submit_button("保存")
        # "保存"ボタンが押されたら
        if submitted:
            if csvlogger:
                # 学習過程をcsv形式で保存
                callback = CSVLogger("./logs/training.csv")
                callbacks[type(callback)] = callback
            if earlystopping:
                # fit中に学習が進まなくなったら学習を止める            
                callback = EarlyStopping(monitor="val_loss", patience=0, verbose=0, mode="auto")
                callbacks[type(callback)] = callback
            st.session_state["callbacks"] = callbacks
            st.success("保存しました")
            time.sleep(.5)
            st.rerun()


# -- UI --
st.title("モデルの保存")
st.write("MNISTを使用したモデルの作成を行います")

num_layers = st.slider("層の数", min_value=2, max_value=15)
layers, params, can_create = create_layers(num_layers)
is_create_model = st.button("開始", disabled=can_create is not True)

with st.sidebar:
    epochs = st.slider("エポック数", min_value=1, max_value=100, value=10, disabled=is_create_model)
    batch_size = st.selectbox("バッチサイズ", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], index=6, disabled=is_create_model)

    st.space("medium")
    if st.button("学習設定", width="stretch", disabled=is_create_model):
        fit_option()

if is_create_model:
    try:
        with st.spinner("モデルの作成中"):
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

            st.space("large")
            plot_history(history)
            
            st.subheader("テスト用データでの精度")
            col1, col2 = st.columns(2)
            col1.write(f"**Accuracy**: {score[1]}")
            col2.write(f"**Loss**: {score[0]}")
            
            # モデルを一時保存
            model.save("./models/my_model.h5")

        st.success("モデルの作成完了", icon="✅")
        st.space("medium")

        col1, col2, col3 = st.columns(3)
        # モデルのダウンロード処理
        with open("./models/my_model.h5", "rb") as f:
                col1.download_button("モデルのダウンロード",
                                data=f,
                                file_name="model.h5",
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


