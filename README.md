# ArcFace
<hr>

## 目的
就活に使う作品のため<br>
思ったよりチューニング云々が難しくて延々と回してる<br>

## 実験環境
|物やライブラリ等|名前/バージョン|
|:-----------|:------------|
|CPU|Ryzen7 3800XT|
|GPU|RTX3090|
|CUDA|11.2|
|Python|3.6|

その他ライブラリはenvironment.txtを参照<br>

## 実行方法
setting.csvの2行目のパラメータのみを実行する場合<br>
`python main.py`<br>
setting.csvのn行目のパラメータを実行する場合<br>
`python main.py n`<br>

## 各ファイルの説明
#### main.py
プログラム全体の実行ファイル<br>
コマンドライン引数を引っ張ってきて何行目のパラメータを呼び出すか指定する<br>
#### train.py
学習を行うTrainerクラスの定義ファイル<br>
#### model.py
各モデルの定義を行うクラスの定義ファイル<br>
#### data_loader.py
データセットの読み込みと前処理を行うDataLoaderクラスの定義ファイル<br>
#### logger.py
ログの保存と保存したログをグラフとして描画し保存するLoggerクラスの定義ファイル<br>
#### parameter_loader.py
setting.csvからパラメータを読み込む関数の定義ファイル<br>
#### setting.csv
モデル名，データセットの場所，ログの保存先，各種ハイパーパラメータを記述しておくcsvファイル<br>
#### multirun.sh
複数パラメータによる実験を行う際に使用<br>

```
for ((i=0 ; i<10 ; i++))
do
  python main.py $i
done
```

ループ回数を変更して回す回数と行を指定
