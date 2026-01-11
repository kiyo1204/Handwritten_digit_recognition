import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical, plot_model
from keras.callbacks import LambdaCallback
from sklearn.model_selection import train_test_split

def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc)+1)

    compare_acc = plt.figure()
    plt.title("Comparison of Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(epochs, acc, "-r")
    plt.plot(epochs, val_acc, "-b")
    plt.grid(True)

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


st.title("モデルの保存")
st.write("MNISTを使用したモデルの作成を行います")
with st.sidebar:
    epochs = st.slider("エポック数", min_value=1, max_value=100, value=10)
    batch_size = st.selectbox("バッチサイズ", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], index=3)

num_layers = st.slider("層の数", min_value=2, max_value=10)
layers = []
params = [[[] for i in range(3)] for i in range(num_layers)]

for layer in range(num_layers):
    layer_type = st.selectbox(f"**{layer+1}層目**", ["Dense", "Conv2D", "Flatten", "MaxPooling2D", "Dropout"])
    param = params[layer]
    if layer_type == "Conv2D":
        col1, col2, col3 = st.columns(3)
        param[0] = col1.selectbox("*Filters(フィルタの数)*", [16, 32, 64, 128, 256, 512], key="filter_"+str(layer))
        param[1] = col2.number_input("*Kernel Size(フィルタの寸法)(N x N)*", min_value=1, max_value=5, key="kernel_"+str(layer))
        param[2] =col3.selectbox("*活性化関数*", ["relu", "softmax"], key="act_cnn_"+str(layer))
        layers.append([layer_type, layer])
    elif layer_type == "Dense":
        col1, col2, col3 = st.columns(3)
        param[0] = col1.number_input("*ユニット数*", min_value=2, max_value=512, key="units_"+str(layer))
        param[1] = col2.selectbox("*活性化関数*", ["relu", "softmax"], key="act_dense_"+str(layer))
        param[2] = None
        layers.append([layer_type, layer])
    elif layer_type == "MaxPooling2D":
        col1, col2, col3 = st.columns(3)
        param[0] = col1.number_input("*プーリングのサイズ(N x N)*", min_value=1, max_value=10, key="pooling_"+str(layer))
        param[1] = None
        param[2] = None
        layers.append([layer_type, layer])
    elif layer_type == "Flatten":
        layers.append([layer_type, layer])
    elif layer_type == "Dropout":
        param[0] = st.slider("ドロップアウト率(出力を減らす)", min_value=0.00, max_value=1.00, step=.01, key="dropout_"+str(layer))
        layers.append([layer_type, layer])

can_create = False
last_activation = params[-1][:2]
first_activation = layers[0][0]

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

is_create_model = st.button("開始", disabled=can_create is not True)

if is_create_model:
    try:
        with st.spinner("モデルの作成中"):
            log_area = st.empty()
            metrics_data = []

            Mnist = tf.keras.datasets.mnist
            (X_train, y_train), (X_val, y_val) = Mnist.load_data()
            y_train, y_val = to_categorical(y_train), to_categorical(y_val)

            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=42)

            model = Sequential()

            for layer, param in zip(layers, params):
                if layer[0] == "Conv2D" and layer[1] == 0:
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


            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            tb_cb = tf.keras.callbacks.TensorBoard(log_dir="/logs", histogram_freq=0, write_graph=True)
            csv_logger = tf.keras.callbacks.CSVLogger("./logs/training.csv")
            es_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=0, verbose=0, mode="auto")
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

            history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(X_test, y_test),
                        callbacks=[tb_cb, csv_logger, es_cb, stream_it_callback]
                    )
            

            score = model.evaluate(X_val, y_val, verbose=1)

            plot_history(history)
            st.write(f"Loss: {score[0]}, Accuracy: {score[1]}")
            plot_model(model, show_shapes=True, to_file="./models/model.png")
            st.image("./models/model.png", width="content")
            model.save("./models/my_model.h5")

        st.success("モデルの保存完了")

        col1, col2, col3 = st.columns(3)
        with open("./models/my_model.h5", "rb") as f:
            col1.download_button("モデルのダウンロード",
                                data=f,
                                file_name="model.h5"
                            )
        with open("./logs/training.csv", "rb") as f:
            col2.download_button("精度の詳細のダウンロード(csv)",
                                data=f,
                                file_name="train_log.csv"
                            )
        

    except Exception as e:
        st.error("モデル作成でエラーが発生しました.\nパラメータの確認をしてください")
        st.subheader("エラー内容")
        st.write(e)



