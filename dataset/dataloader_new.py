from torch.utils.data import Dataset
import tqdm
import json
import torch
import random
import numpy as np
from sklearn.utils import shuffle
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from feature import mol2alt_sentence
from rdkit.Chem import MolStandardize

''''
应用于合并
'''
# 数据预处理和增强部分
class CLSDataset(Dataset):
    def __init__(self, corpus_path, word2idx, max_seq_len,
                 dataset_name="HIV", data_regularization=False):

        self.data_regularization = data_regularization

        # define max length
        self.max_seq_len = max_seq_len
        # directory of corpus dataset
        self.corpus_path = corpus_path
        # define special symbols
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        self.dataset_name = dataset_name
        self.word2idx = word2idx
        self.unk_nums = 0

        # 加载语料
        with open(corpus_path, "r", encoding="utf-8") as f:
            # 将数据集全部加载到内存
            self.lines = [line for line in tqdm.tqdm(f, desc="Loading Dataset")]
            # 打乱顺序
            self.lines = shuffle(self.lines)
            # 获取数据长度(条数)
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # 得到tokenize之后的文本和与之对应的分类
        text, label = self.get_text_and_label(item) # 获得对应的文本和标签

        text_input = self.tokenize_char(text)

        # 添加#CLS#和#SEP#特殊token
        text_input = [self.cls_index] + text_input + [self.sep_index]
        # 如果序列的长度超过self.max_seq_len限定的长度, 则截断
        text_input = text_input[:self.max_seq_len]

        output = {"text_input": torch.tensor(text_input),
                  "label": torch.tensor([label])}
        return output

    def standardizeAndcanonical(self,smi):
        lfc = MolStandardize.fragment.LargestFragmentChooser()
        # standardize
        mol = Chem.MolFromSmiles(smi)
        mol2 = lfc.choose(mol)
        smi2 = Chem.MolToSmiles(mol2)
        return smi2

    def get_text_and_label(self, item):
        # 获取文本和标记

        if self.dataset_name == "hiv" or self.dataset_name == "bbbp" \
                or self.dataset_name == 'bace' or self.dataset_name == 'estrogen-alpha'\
            or self.dataset_name == 'estrogen-alpha' or self.dataset_name == 'estrogen-beta'\
                or self.dataset_name == 'mesta-high' or self.dataset_name == 'mesta-low':
            line = self.lines[item].split(",")
            # print(line)
            smiles = line[0]
            if smiles[-1] == "\n":
                smiles = smiles[:-1]
            smiles = self.standardizeAndcanonical(smiles)
            t = Chem.MolFromSmiles(smiles)
            sentence_0 = mol2alt_sentence(t, 0)
            sentence_1 = mol2alt_sentence(t, 1)
            sentence = []
            l = len(sentence_0)
            # if len(sentence_1) % 2 != 0:
                # print(1)
            sentence = sentence_1[l:]
            label = int(line[1][:1])
            # print(label)
        elif self.dataset_name == "freesolv" or self.dataset_name == 'esol' or \
                self.dataset_name == 'lipophilicity':
            line = self.lines[item].split(",")
            # print(line)
            smiles = line[0]
            if smiles[-1] == '\n':
                smiles = smiles[:-1]
            if smiles[-1] == ' ':
                smiles = smiles[:-1]
            # print(smiles)
            smiles = self.standardizeAndcanonical(smiles)
            t = Chem.MolFromSmiles(smiles)
            sentence_0 = mol2alt_sentence(t, 0)
            sentence_1 = mol2alt_sentence(t, 1)
            l = len(sentence_0)
            sentence = sentence_1[l:]

            # adding = 3.0 if self.dataset_name == 'esol' else 6.0
            if self.dataset_name == 'lipophilicity':
                # print(123)
                adding = 2.0
            elif self.dataset_name == 'esol':
                adding = 12.0
            elif self.dataset_name == 'freesolv':
                adding = 6.0
            # line[1] = line[1][:-1] if line[1][-1] == '\n' else line[1]
            label = float(line[1]) + adding
            # print(label,line[1])
            # print(label)
        else:
            # 多任务
            line = self.lines[item].split(",")
            smiles = line[0]
            label = []
            for lab in line[1:]:
                label.append(int(lab))
            # print(label)
            if smiles[-1] == '\n':
                smiles = smiles[:-1]
            if smiles[-1] == ' ':
                smiles = smiles[:-1]
            smiles = self.standardizeAndcanonical(smiles)
            t = Chem.MolFromSmiles(smiles)
            sentence_0 = mol2alt_sentence(t, 0)
            sentence_1 = mol2alt_sentence(t, 1)
            l = len(sentence_0)
            sentence = sentence_1[l:]

        return sentence, label

    def tokenize_char(self, segments):
        return [self.word2idx.get(char, self.unk_index) for char in segments]