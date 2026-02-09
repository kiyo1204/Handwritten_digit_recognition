# MNISTを用いた手書き数字認識アプリ
URL: https://handwrittendigitrecognition-4ckfcz9af5ujpdizjuuuus.streamlit.app/

## 概要(Description)
このアプリは手書き数字のデータセットであるMNISTを使用し, オリジナルなモデルを作成・精度の確認を行うことができます. フレームワークにはStreamlitを使用し, Streamlit Cloudで公開しています.

## 使用例(Example)
- ### 手書き数字の予測
https://github.com/user-attachments/assets/d3753351-94cc-4724-8e50-f49d110ee17b

- ### モデルの作成
https://github.com/user-attachments/assets/549f728d-ce0e-430d-b37b-568f254197b7

## 実行環境
- **言語**
  - Python 3.11

- **主要ライブラリ**
  - Streamlit
  - Tensorflow / Keras
  - Matplotllib
 
  そのほかのライブラリやバージョンは*requirements.txt*に記載されています.

## コードのインストール方法(Installation)
このアプリはクラウドにアップロードしているため使用にダウンロードの必要はありませんが, 自分で機能を追加したりしたいときはダウンロードして追加できます.

1. 保存したいディレクトリに移動する.
2. 右クリックでターミナルを起動する(**保存したいディレクトリがカレントディレクトリになっていることを確認**).
   <img width="1135" height="505" alt="Install_file" src="https://github.com/user-attachments/assets/ff7fbe69-e074-4375-9543-624b7d65667f" />
3. レポジトリのデータをダウンロード.
   ```bash
   git clone https://github.com/kiyo1204/Handwritten_digit_recognition
   ```
4. 必要なモジュールのインストール
   ```bash
   pip install -r requirements.txt
   ```
   
## 使い方(Usage)
手書き数字の予測を行うためにはモデルのファイル(h5ファイル)が必要です. モデルファイルは**モデルの保存**ページで作成できます.

### モデルの作成
例として簡単なニューラルネットワークの作成を行います.
1. 最初の層は**平坦化**を行います.<br>
   平坦化(Flatten)とは, 多次元の配列を1次元に変換することです. モデル作成時のデータの入力は28×28×1の3次元の画像データなので全結合層の前に平坦化(Flatten)を行って1次元のデータに変換する必要があります(コード内でこの変換は省けますが, データの形を理解するためにあえて省いていません).
   入力画像の例は以下のようになります.
   <img width="622" height="707" alt="image" src="https://github.com/user-attachments/assets/21a36886-23d2-484a-885f-c8b0c5a30733" />

3. 次の層に**全結合層**を入れます.<br>
   全結合層(Dense)は, 1つの入力ユニットが次の層のユニットすべてに繋がっている層です. ユニット数はお好みですが, 今回は128とします. また, 活性化関数はReLUとします.

4. 最終層も全結合層を入れ, ユニット数は10で活性化関数はソフトマックスとします.<br>
   構成は以下のようになります.
   <img width="1095" height="818" alt="image" src="https://github.com/user-attachments/assets/c9fe00b0-4711-45e8-b930-3dad63cf53d9" />
5. **開始**を押します.<br>
   問題がなく学習できればエラーが発生せずにモデルの精度や損失のグラフが表示されます. その下の**モデルの作成完了**欄をクリックするとモデルのダウンロードボタンが表示されます.
   <img width="1060" height="746" alt="image" src="https://github.com/user-attachments/assets/8bf4e6ea-3665-4e42-9db5-7ce1b90c34a9" />
   >ここでの検証用データとテスト用データの違いは, 検証用データはモデルの学習をコントロールするためのデータ(Early Stoppingなど)で, テスト用データはモデル学習後に1回だけ評価を行うためのデータです. また, 検証用データへの過学習を防ぐためにも分けています.

  
### 手書き数字の予測
1. ダウンロードしたモデルのファイルをアップロードします.
2. 手書き欄で書いてみてエラーが発生しなければ成功です.

この例では簡単なニューラルネットワークを作成しましたが, このアプリでは畳み込み層やプーリング層にも対応しているためCNN(Convolutional Neural Network)も作成できます.
