import os
import numpy as np
import torch
from torch import Tensor
import librosa
from torch.utils.data import Dataset


# Audioni padding qilish
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # Padding kerak
    num_repeats = (max_len // x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, cut=64600):
        """
        Args:
            list_IDs: Utts kalitlari ro'yxati (string).
            labels: Kalitlar va tegishli yorliqlar lug'ati.
            base_dir: Ma'lumotlar joylashgan katalog (flac katalogsiz).
            cut: Maksimal uzunlik (standart: 64600).
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = cut

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        file_path = os.path.join(self.base_dir, f"{key}.flac")  # flac ni qayta qo‘shmang
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        X, fs = librosa.load(file_path, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


# ASVspoof2021 baholash ma'lumotlar to'plami uchun Dataset sinfi
class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir, cut=64600):
        self.list_IDs = [x.replace(' ', '_') for x in list_IDs]  # Bo'sh joylarni almashtirish
        self.base_dir = base_dir
        self.cut = cut

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        file_path = os.path.join(self.base_dir, f"{key}.flac")
        if not os.path.exists(file_path):
            print(f"Checking file: {file_path}, Exists: {os.path.exists(file_path)}")  # Fayl mavjudligini tekshirish
            raise FileNotFoundError(f"File not found: {file_path}")

        X, fs = librosa.load(file_path, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key

