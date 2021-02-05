from train import Trainer   # 自作(学習に使用)
import os                   # ファイルの存在確認に使用
import sys                  # コマンドライン引数の読み取りに使用

# main関数
def main():
    index = 0           # Trainerへ送るindexを初期化
    args = sys.argv     # コマンドライン引数を取得
    # 1つ目の引数が存在しなければindexは初期値とする
    # 1つ目の引数が数字でなければindexは初期値とする
    if len(args) > 1:
        if args[1].replace(',', '').replace('.', '').replace('-', '').isdigit():    # 余計なものを消してから判定(小数を含む場合は小数点が消されるため小数点を消した数になる)
            index = int(args[1])
        else:
            # だいたい見た目通りの出力がされる
            print("----------------------------------")
            print("")
            print("Argument is not digit")
            print("Set index to 0")
            print("")
            print("----------------------------------")
    else:
        # だいたい見た目通りの出力がされる
        print("----------------------------------")
        print("")
        print("Arguments are too short")
        print("Set index to 0")
        print("")
        print("----------------------------------")
    setting_csv_path = "./setting.csv"                                  # パラメータが保存されたcsvファイルのパスを指定
    trainer = Trainer(setting_csv_path=setting_csv_path, index=index)   # 学習のうんぬんかんぬんを管理するクラスを変数として宣言
    # 学習済みファイルが存在しない場合のみ学習を行う(パラメータチューニングの際に既に試したパラメータをスキップするため)
    if not os.path.isfile(os.path.join(trainer.log_path, trainer.model_name)):
        print("Trained weight file does not exist")
        trainer.train()

if __name__ == "__main__":
    main()
