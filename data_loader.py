import os
import numpy as np
import joblib
import csv
import torch
from torch import utils
from torchvision import datasets, transforms
from PIL import Image

class DataLoader:
    def __init__(self, data_path, batch_size, img_size, train_ratio=0.8, num_workers=0, pin_memory=True):
        self.dataset_path = os.path.join(data_path)                  #画像データのパス
        self.img_size = img_size
        self.batch_size = batch_size                #バッチサイズ
        #画像の変換方法の選択
        self.train_transform = transforms.Compose([transforms.Resize(self.img_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(3),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.validation_transform = transforms.Compose([transforms.Resize(self.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        #self.load_data()
        self.dataloaders = self.import_image(batch_size=batch_size, train_ratio=train_ratio, num_workers=num_workers, pin_memory=True)

#学習に使うデータの読み込みを行う関数
#Argument
#dataset_path: データセットが格納されているディレクトリのパス
#batch_size: バッチサイズ
#train_ratio: データ全体から学習に使うデータの割合(デフォルト値: 0.8)
#img_size: 画像のサイズ タプルで指定すること
#
    def import_image(self, batch_size, train_ratio=0.8, num_workers=0, pin_memory=True):
        #torchvision.datasets.ImageFolderで画像のディレクトリ構造を元に画像読み込みとラベル付与を行ってくれる
        #transformには前処理を記述
        data = datasets.ImageFolder(root=self.dataset_path, transform=self.train_transform)
        #print(data.class_to_idx)

        train_size = int(train_ratio * len(data))           #学習データ数
        val_size = len(data) - train_size                   #検証データ数
        data_size = {"train":train_size, "val":val_size}    #それぞれのデータ数をディクショナリに保存

        train_data, val_data = utils.data.random_split(data, [train_size, val_size])    #torcn.utils.data.random_splitで重複なしにランダムな分割が可能


        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)   #学習データのデータローダー
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)      #検証データのデータローダー
        dataloaders = {"train":train_loader, "val":val_loader}                                  #それぞれのデータローダーをディクショナリに保存
        #datas = {"train":train_data, "val":val_data}
        return dataloaders#, datas


class MakeDataLoader(utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.targets = []
        print("Making target list...")
        for i in range(len(data)):
            self.targets.append(data[i][1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index][0], self.targets[index]
        return img, target
