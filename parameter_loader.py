import csv              # csvファイルを読み込むために使用
from tqdm import tqdm   # 読み込みの進捗を可視化するために使用

# csvファイルから各種パラメータを一括読み込みする関数
# Argument
# csv_path  : csvファイルのパス
# index     : 何行目を使用するか
def read_parameters(csv_path, index):
    with open(csv_path, encoding='utf-8-sig') as f: #utf-8-sigでエンコードしないと1列目のキーがおかしくなる
        reader = csv.DictReader(f)          # 辞書型で読み込み
        l = [row for row in tqdm(reader)]   # 行ごとに格納
        parameters_dict = l[index]          # 必要な行のみを選択

    return parameters_dict

#文字列のTrueをbool値のTrueに変換しそれ以外をFalseに変換する関数
def str_to_bool(str):
    return str.lower() == "true"
