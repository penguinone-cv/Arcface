import torch                                                                        # pytorchを使用(モデル定義及び学習フロー等)
import torch.nn as nn                                                               # torchライブラリ内のnnパッケージ
import numpy as np                                                                  # numpy(行列計算に使用)
import os                                                                           # パス作成とCPUのコア数読み込みに使用
import math
from pytorch_metric_learning import losses                                          # Loss関数の呼び出しに使用
from tqdm import tqdm                                                               # 学習の進捗表示に使用
from model import VGG_based, ResNet_based, PretrainedResNet                         # 自作(モデル定義に使用)
from logger import Logger                                                           # 自作(ログ保存に使用)
from data_loader import DataLoader                                                  # 自作(データ読み込みに使用)
from parameter_loader import read_parameters, str_to_bool                           # 自作(パラメータ読み込みに使用)

# 学習全体を管理するクラス
class Trainer:
    def __init__(self, setting_csv_path, index):
        self.parameters_dict = read_parameters(setting_csv_path, index)                             # 全ハイパーパラメータが保存されたディクショナリ
        self.model_name = self.parameters_dict["model_name"]                                        # モデル名
        log_dir_name = self.model_name + "_epochs" + self.parameters_dict["epochs"] \
                            + "_batch_size" + self.parameters_dict["batch_size"] \
                            + "_lr" + self.parameters_dict["learning_rate"] \
                            + "_margin" + self.parameters_dict["margin"] \
                            + "_scale" + self.parameters_dict["scale"]                              # ログを保存するフォルダ名
        self.log_path = os.path.join(self.parameters_dict["base_log_path"], log_dir_name)           # ログの保存先
        batch_size = int(self.parameters_dict["batch_size"])                                        # バッチサイズ
        learning_rate = float(self.parameters_dict["learning_rate"])                                # 学習率
        momentum = float(self.parameters_dict["momentum"])                                          # 慣性項(SGD使用時のみ使用)
        weight_decay = float(self.parameters_dict["weight_decay"])                                  # 重み減衰(SGD使用時のみ使用)
        img_size = (int(self.parameters_dict["width"]),int(self.parameters_dict["height"]))         # 画像サイズ
        self.logger = Logger(self.log_path)                                                         # ログ書き込みを行うLoggerクラスの宣言
        num_class = int(self.parameters_dict["num_class"])                                          # クラス数
        step_size = int(self.parameters_dict["step_size"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                  # GPUが利用可能であればGPUを利用
        #self.model = VGG_based(num_class).to(self.device)                                          # ネットワークを定義(VGGベース)
        #self.model = ResNet_based(num_class).to(self.device)                                       # ネットワークを定義(ResNetベース)
        self.model = PretrainedResNet(num_class).to(self.device)                                    # ネットワークを定義(学習済みResNetベース)
        #学習済み重みファイルがあるか確認しあれば読み込み
        if os.path.isfile(os.path.join(self.log_path, self.model_name)):
            print("Trained weight file exists")
            self.model.load_state_dict(torch.load(os.path.join(self.log_path, self.model_name)))

        #CNN部分の最適化手法の定義
        #ArcFaceLoss
        #簡単のためpytorch_metric_learningからimportして読み込み
        #margin : クラス間の分離を行う際の最少距離(cosine類似度による距離学習を行うためmarginはθを示す)
        #scale : クラスをどの程度の大きさに収めるか
        #num_classes : ArcFaceLossにはMLPが含まれるためMLPのパラメータとして入力
        #embedding_size : 同上
        self.loss = losses.ArcFaceLoss(margin=float(self.parameters_dict["margin"]),
                                        scale=int(self.parameters_dict["scale"]),
                                        num_classes=num_class,
                                        embedding_size=512).to(self.device)
        #self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.loss.parameters()), lr=self.learning_rate)      # PMLのArcFaceLossにはMLPが含まれている(Trainable)なのでモデルパラメータとlossに含まれるモデルパラメータを最適化
        self.optimizer = torch.optim.SGD(list(self.model.parameters()) + list(self.loss.parameters()), lr=learning_rate,
                                            momentum=momentum, weight_decay=weight_decay, nesterov=True)                  # PMLのArcFaceLossにはMLPが含まれている(Trainable)なのでモデルパラメータとlossに含まれるモデルパラメータを最適化
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=0.1)


        #バッチ読み込みをいくつのスレッドに並列化するか指定
        #パラメータ辞書に"-1"と登録されていればCPUのコア数を読み取って指定
        num_workers = 0
        if int(self.parameters_dict["num_workers"]) == -1:
            print("set num_workers to number of cpu cores :", os.cpu_count())
            num_workers = os.cpu_count()
        else:
            num_workers = int(self.parameters_dict["num_workers"])

        #データローダーの定義
        #data_path : データの保存先
        #batch_size : バッチサイズ
        #img_size : 画像サイズ(タプルで指定)
        #train_ratio : 全データ中学習に使用するデータの割合
        self.data_loader = DataLoader(data_path=self.parameters_dict["data_path"],
                                      batch_size=int(self.parameters_dict["batch_size"]),
                                      img_size=img_size,
                                      train_ratio=float(self.parameters_dict["train_ratio"]),
                                      num_workers=num_workers, pin_memory=str_to_bool(self.parameters_dict["pin_memory"]))


    # 学習を行う関数
    def train(self):
        torch.backends.cudnn.benchmark = True                   # 学習の計算再現性が保証されない代わりに高速化を図る(研究用途でないためTrueとする)
        print("Train phase")
        print("Train", self.model_name)

        epochs = int(self.parameters_dict["epochs"])            # epoch数

        # 学習ループ開始(tqdmによって進捗を表示する)
        with tqdm(range(epochs)) as pbar:
            for epoch in enumerate(pbar):
                i = epoch[0]                                    # 現在のepoch
                pbar.set_description("[Epoch %d]" % (i+1))      # プログレスバーのタイトル部分の表示を変更
                loss_result = 0.0                               # Lossを保存しておく
                acc = 0.0                                       # Accuracyの計算に使用
                val_loss_result = 0.0                           # Validation Lossを保存しておく
                val_acc = 0.0                                   # Validation Accuracyの計算に使用

                self.model.train()                              # モデルをtrainモード(重みが変更可能な状態)にする
                j = 1                                           # 現在のiterationを保存しておく変数
                for inputs, labels in self.data_loader.dataloaders["train"]:            # イテレータからミニバッチを順次読み出す
                    pbar.set_description("[Epoch %d (Iteration %d)]" % ((i+1), j))      # 現在のiterationをプログレスバーに表示
                    inputs = inputs.to(self.device, non_blocking=True)                  # 入力データをGPUメモリに送る(non_blocking=Trueによってasynchronous GPU copiesが有効になりCPUのPinned MemoryからGPUにデータを送信中でもCPUが動作できる)
                    labels = labels.clone().detach()
                    labels = labels.to(self.device, non_blocking=True)                  # 教師ラベルをGPUメモリに送る
                    outputs = self.model(inputs)                                        # モデルにデータを入力し出力を得る
                    #print(outputs.shape())
                    loss = self.loss(outputs, labels)                                   # 教師ラベルとの損失を計算

                    # ArcFaceLossから出力を取り出してAccuracyを計算する
                    # 参考: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/175
                    mask = self.loss.get_target_mask(outputs, labels)
                    cosine = self.loss.get_cosine(outputs)
                    cosine_of_target_classes = cosine[mask == 1]
                    modified_cosine_of_target_classes = self.loss.modify_cosine_of_target_classes(cosine_of_target_classes, cosine, outputs, labels, mask)
                    diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
                    logits = cosine + (mask*diff)
                    logits = self.loss.scale_logits(logits, outputs)
                    pred = logits.argmax(dim=1, keepdim=True)
                    for img_index in range(len(labels)):
                        if pred[img_index, 0] == labels[img_index]:
                            acc += 1.

                    self.optimizer.zero_grad()
                    loss.backward()

                    self.optimizer.step()

                    #_, preds = torch.max(outputs, 1)
                    loss_result += loss.item()
                    #acc += torch.sum(preds == labels.data)


                    j = j + 1

                else:
                    with torch.no_grad():
                        self.model.eval()
                        pbar.set_description("[Epoch %d (Validation)]" % (i+1))
                        for val_inputs, val_labels in self.data_loader.dataloaders["val"]:
                            val_inputs = val_inputs.to(self.device, non_blocking=True)
                            val_labels = val_labels.clone().detach()
                            val_labels = val_labels.to(self.device, non_blocking=True)
                            val_outputs = self.model(val_inputs)
                            val_loss = self.loss(val_outputs, val_labels)

                            #_, val_preds = torch.max(val_outputs, 1)
                            val_loss_result += val_loss.item()
                            #val_acc += torch.sum(val_preds == val_labels.data)
                            mask = self.loss.get_target_mask(val_outputs, val_labels)
                            cosine = self.loss.get_cosine(val_outputs)
                            cosine_of_target_classes = cosine[mask == 1]
                            modified_cosine_of_target_classes = self.loss.modify_cosine_of_target_classes(cosine_of_target_classes, cosine, val_outputs, val_labels, mask)
                            diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
                            logits = cosine + (mask*diff)
                            logits = self.loss.scale_logits(logits, val_outputs)
                            pred = logits.argmax(dim=1, keepdim=True)
                            for img_index in range(len(val_labels)):
                                if pred[img_index, 0] == val_labels[img_index]:
                                    val_acc += 1.

                    epoch_loss = loss_result / len(self.data_loader.dataloaders["train"].dataset)
                    epoch_acc = acc / len(self.data_loader.dataloaders["train"].dataset)
                    val_epoch_loss = val_loss_result / len(self.data_loader.dataloaders["val"].dataset)
                    val_epoch_acc = val_acc / len(self.data_loader.dataloaders["val"].dataset)
                    self.logger.collect_history(loss=epoch_loss, accuracy=epoch_acc, val_loss=val_epoch_loss, val_accuracy=val_epoch_acc)
                    self.logger.writer.add_scalars("losses", {"train":epoch_loss,"validation":val_epoch_loss}, (i+1))
                    self.logger.writer.add_scalars("accuracies", {"train":epoch_acc, "validation":val_epoch_acc}, (i+1))
                    self.logger.writer.add_scalars("learning_rate", {"learning_rate":self.optimizer.param_groups[0]['lr']}, (i+1))
                    self.scheduler.step(val_epoch_loss)

                pbar.set_postfix({"loss":epoch_loss, "accuracy": epoch_acc, "val_loss":val_epoch_loss, "val_accuracy": val_epoch_acc})

        torch.save(self.model.state_dict(), os.path.join(self.log_path,self.model_name))
        self.logger.draw_graph()
        self.logger.writer.flush()
