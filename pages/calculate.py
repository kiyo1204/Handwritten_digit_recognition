import streamlit as st
import numpy as np
import tempfile
import re
import random
import time

from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
from io import StringIO
from PIL import Image


# 画像の予測処理（図と予測ラベルを返す）
def predict_image(tens_place, ones_place):
    pred_tens = None
    pred_ones = None

    if tens_place is not None and tens_place.image_data is not None:
        drawn_image = tens_place.image_data
        drawn_image_gray = drawn_image[:, :, 3]
        if np.sum(drawn_image_gray) > 0:
            drawn_image_gray = Image.fromarray(drawn_image_gray)
            resized_image = np.array(drawn_image_gray.resize((28, 28)))
            
            if model is not None:
                predict_data = resized_image.reshape(-1, 28, 28, 1)
                pred = model.predict(predict_data)
                pred_tens = pred.argmax()
        
    if ones_place is not None and ones_place.image_data is not None:
        drawn_image = ones_place.image_data
        drawn_image_gray = drawn_image[:, :, 3]
        if np.sum(drawn_image_gray) > 0:
            drawn_image_gray = Image.fromarray(drawn_image_gray)
            resized_image = np.array(drawn_image_gray.resize((28, 28)))
            
            if model is not None:
                predict_data = resized_image.reshape(-1, 28, 28, 1)
                pred = model.predict(predict_data)
                pred_ones = pred.argmax()

    if pred_tens is not None and pred_ones is not None:
        return int(str(pred_tens) + str(pred_ones))
    elif pred_ones is not None:
        return pred_ones
    else:
        return "判定不可"

def calculate(x, y):
    is_correct = False

    st.write(f"# {x}+{y}")
    # 手書き欄を横並びに配置
    col1, col2 = st.columns(2)

    with col1:
        tens_place = st_canvas(
            stroke_width=20,
            update_streamlit=True,
            height=400,
            width=340,
            drawing_mode="freedraw",
            key="tens_place"
        )
    with col2:
        ones_place = st_canvas(
            stroke_width=20,
            update_streamlit=True,
            height=400,
            width=340,
            drawing_mode="freedraw",
            key="ones_place"
        )

    ans = predict_image(tens_place, ones_place)
    if x+y == ans:
        is_correct = True
    col1, col2 = st.columns(2)
    col1.write(f"## ={ans}")

    if is_correct:
        col2.success("## 正解")
        st.session_state["x"] = random.randint(0, 49)
        st.session_state["y"] = random.randint(0, 50)
        
        with st.spinner("### 次の問題...."):
            time.sleep(1)
            st.rerun()
    else:
        col2.error("## 不正解")

    if st.button("やめる"):
        st.session_state["play_calculate"] = False
        time.sleep(.1)
        st.rerun()

def load(model_file): # モデルの読み込み
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_file:
        tmp_file.write(model_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        model = load_model(tmp_file_path, safe_mode=False)
        st.success("モデル読み込み完了")
        with StringIO() as buf:
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            text = buf.getvalue()
        st.subheader("モデルの詳細")
        text = re.sub(r"[^a-zA-Z0-9().,\n: ]", "", text)
        st.write(text)
        
        return model
    except Exception as e:
        st.error(f"モデルの読み込みでエラーが発生しました: {e}")

# -- UI --
st.title("計算")
st.write("作成したモデルを使用して, 簡単な計算をしてみましょう")

st.subheader("モデルのアップロード")
model_file = st.file_uploader("ファイルをアップロードしてください", ["keras"])
if model_file:
    model = load(model_file)
else:
    model = None
    st.warning("予測にはモデルの保存をしてください")

st.space("medium")
if st.button("始める", disabled=model is None):
    st.session_state["play_calculate"] = True
    st.session_state["x"] = random.randint(0, 50)
    st.session_state["y"] = random.randint(0, 49)

if st.session_state["play_calculate"]:
    calculate(st.session_state["x"], st.session_state["y"])