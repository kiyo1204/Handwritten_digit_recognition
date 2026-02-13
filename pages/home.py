import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tempfile
import re

from streamlit_drawable_canvas import st_canvas
from PIL import Image
from keras.models import load_model
from io import StringIO

# 画像の予測処理（図と予測ラベルを返す）
def predict_image(canvas_result):
    if canvas_result is not None and canvas_result.image_data is not None:
        drawn_image = canvas_result.image_data
        drawn_image_gray = drawn_image[:, :, 3]
        if np.sum(drawn_image_gray) > 0:
            drawn_image_gray = Image.fromarray(drawn_image_gray)
            resized_image = np.array(drawn_image_gray.resize((28, 28)))
            fig = plt.figure()
            pred_label = None
            if model is not None:
                predict_data = resized_image.reshape(-1, 28, 28, 1)
                pred = model.predict(predict_data)
                pred_label = int(pred.argmax())
            plt.imshow(resized_image, cmap="gray")
            plt.axis("off")
            return fig, pred_label
    return None, None

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
model_file = st.file_uploader("ファイルをアップロードしてください", ["keras"])
if model_file:
    # モデルの読み込み
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
    except Exception as e:
        st.error(f"モデルの読み込みでエラーが発生しました: {e}")
else:
    model = None
    st.warning("予測にはモデルの保存をしてください")

st.space("medium")

# 手書き欄と表示欄を横並びに配置
col1, col2 = st.columns(2)

with col1:
    st.subheader("文字の手書き")
    canvas_result = st_canvas(
        stroke_width=pens_width,
        update_streamlit=True,
        height=400,
        width=340,
        drawing_mode="freedraw",
    )

with col2:
    st.subheader("認識結果")
    fig, pred_label = predict_image(canvas_result)
    if fig is not None:
        st.pyplot(fig)
        if pred_label is not None:
            st.metric("予測", str(pred_label))
    else:
        st.info("キャンバスに数字を描くと、ここに認識結果が表示されます。")

col1, col2 = st.columns(2)
if col1.button("Kerasについて"):
    show_markdown()