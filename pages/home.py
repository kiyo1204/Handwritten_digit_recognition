import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tempfile

from streamlit_drawable_canvas import st_canvas
from PIL import Image
from keras.models import load_model
from io import StringIO

# 画像の予測処理
def predict_image(canvas_result):
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
                    predict_data = resized_image.reshape(-1, 28, 28, 1)
                    pred = model.predict(predict_data)
                plt.title(pred.argmax())

            plt.imshow(resized_image)
            st.pyplot(fig, width="stretch")

@st.dialog("説明")
def show_markdown():
    st.markdown("""
## Tensorflowについて
TensorflowはGoogleが開発したオープンソースのフレームワークであり, *Keras*を高レベルAPIとしてサポートしている.

## Kerasについて
KerasはPythonで書かれたオープンソースのディープラーニングライブラリである.<br>直観的に使えるため, 複雑なモデルを簡単に構築できる他, 拡張性にも優れるため幅広いユーザーに適す.
### 主な特徴
- 直観的なAPI: 複雑なモデルを数行で構築できる.
- 高い拡張性: 独自のレイヤーや損失関数を使用できる.
- クロスプラットフォーム対応: CPUやGPUといった異なる計算リソースに対応している.

## MNISTについて
MNISTは0～9までの手書き数字を集めたデータセットで, 学習用で60,000枚, テスト用で10,000枚用意されている.<br>また, 画像のサイズは28×28ピクセルで色はグレースケールになっている.

## 基本的な学習の流れ
1. **モデルの定義**
    - 層(layer)を積み重ねてネットワークを構築.
2. **コンパイル**
    - 最適化アルゴリズムや損失関数の定義.
3. **学習**
    - 訓練データを使用し, モデルの構築.
4. **評価・予測**
    - テストデータでモデルの制度を確認したり, 未知データの予測を行う.
""", unsafe_allow_html=True)


# -- UI --
st.title("MNISTを使った数字認識")
st.space("small")

# サイドバー設定
with st.sidebar:
    pens_width = st.slider("ペンの太さ", min_value=10, max_value=100, value=40)

st.subheader("モデルのアップロード")
model_file = st.file_uploader("ファイルをアップロードしてください", ["h5"])
if model_file:
    # モデルの読み込み
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
        tmp_file.write(model_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        model = load_model(tmp_file_path)
        st.success("モデル読み込み完了")
        with StringIO() as buf:
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            text = buf.getvalue()
        st.subheader("モデルの詳細")
        st.write(text)
    except Exception as e:
        st.error("モデルの読み込みでエラーが発生しました: {e}")
else:
    model = None
    st.warning("予測にはモデルの保存をしてください")

st.space("medium")

# 手書き欄
st.subheader("文字の手書き")
canvas_result = st_canvas(
    stroke_width=pens_width,
    update_streamlit=True,
    height="400",
    width="400",
    drawing_mode="freedraw",
)
predict_image(canvas_result)

col1, col2 = st.columns(2)
if col1.button("Kerasについて"):
    show_markdown()