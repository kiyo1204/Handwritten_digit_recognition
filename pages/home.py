import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tempfile

from streamlit_drawable_canvas import st_canvas
from PIL import Image
from keras.models import load_model 

st.title("MNISTを使った数字認識")
st.space("small")

st.subheader("モデルのアップロード")
model_file = st.file_uploader("ファイルをアップロードしてください", ["h5"])
if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
        tmp_file.write(model_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        model = load_model(tmp_file_path)
        st.success("モデル読み込み完了")
    except Exception as e:
        st.error("モデルの読み込みでエラーが発生しました.{e}")
else:
    model = None
    st.warning("予測にはモデルの保存をしてください")

pens_width = st.sidebar.slider("ペンの太さ", min_value=10, max_value=100, value=40)

st.write("文字の手書き")
canvas_result = st_canvas(
    stroke_width=pens_width,
    update_streamlit=True,
    height="400",
    width="400",
    drawing_mode="freedraw",
)

if canvas_result.image_data is not None:
    drawn_image = canvas_result.image_data
    drawn_image_gray = drawn_image[:, :, 3]
    if np.sum(drawn_image_gray) > 0:
        drawn_image_gray = Image.fromarray(drawn_image_gray)
        resized_image = np.array(drawn_image_gray.resize((28, 28)))
        
        fig = plt.figure()
        if model is not None:
            try:
                predict_data = resized_image.reshape(-1, 28, 28, 1)
                pred = model.predict(predict_data)
            except ValueError:
                predict_data = resized_image.reshape(-1, 784)
                pred = model.predict(predict_data)
            plt.title(pred.argmax())

        plt.imshow(resized_image)
        st.pyplot(fig, width="stretch")
        

