from torch.utils.data import Dataset
import tqdm
import json
import torch
import random
import numpy as np
import pickle
from sklearn.utils import shuffle
from rdkit import Chem
from rdkit.Chem import AllChem
from feature import mol2alt_sentence
from rdkit.Chem import MolStandardize


'''
不合并
'''
# 基于smiles格式数据，没有上下句连贯，故只用msl来训练语言模型。
class BERTDataset(Dataset):
    def __init__(self, corpus_path, word2idx_path, seq_len, hidden_dim=300, on_memory=True):
        # hidden dimension for positional encoding
        self.hidden_dim = hidden_dim

        # define path of dicts
        self.word2idx_path = word2idx_path

        # define max length
        self.seq_len = seq_len

        # load whole corpus at once or not
        self.on_memory = on_memory

        # directory of corpus dataset
        self.corpus_path = corpus_path

        # define special symbols
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4

        # # 加载字典
        #with open(word2idx_path, "r", encoding="utf-8") as f:
        #     self.word2idx = json.load(wf)

        # 加载语料
        ident = open(word2idx_path, 'rb')
        self.ident_dict = pickle.load(ident)
        print(self.ident_dict)

        # 加载语料
        # smiles 格式是一条string
        with open(corpus_path, "r", encoding="utf-8") as f:
            if not on_memory:
                # 如果不将数据集直接加载到内存, 则需先确定语料行数
                self.corpus_lines = 0
                for _ in tqdm.tqdm(f, desc="Loading Dataset"):
                    self.corpus_lines += 1

            if on_memory:
                # 将数据集全部加载到内存
                self.lines = [line for line in tqdm.tqdm(f, desc="Loading Dataset")]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            # 如果不全部加载到内存, 首先打开语料
            self.file = open(corpus_path, "r", encoding="utf-8")

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1 = self.get_corpus_line(item)

        t1_random, t1_label = self.random_char(t1)
        t1 = [self.cls_index] + t1_random + [self.sep_index]
        t1_label = [self.cls_index] + t1_label + [self.sep_index]

        bert_input = t1[:self.seq_len]
        bert_label = t1_label[:self.seq_len]

        output = {
            "bert_input": torch.tensor(bert_input),
            "bert_label": torch.tensor(bert_label),
        }
        return output

    def tokenize_char(self, segments):
        return [self.ident_dict.get(char, self.unk_index) for char in segments]

    def random_char(self, sentence):
        '''
            sentence : 类型列表[]
                      一系列标识符
        '''
        char_tokens_ = sentence
        char_tokens = self.tokenize_char(char_tokens_)

        output_label = []
        for i, token in enumerate(char_tokens):
            prob = random.random()
            output_label.append(char_tokens[i])
            if prob < 0.30:
                prob /= 0.30
                # 80% randomly change token to mask token
                if prob < 0.8:
                    char_tokens[i] = self.mask_index
                # 10% randomly change token to random token
                elif prob < 0.9:
                    char_tokens[i] = random.randrange(len(self.ident_dict))

        return char_tokens, output_label

    def standardizeAndcanonical(self, smi):
        lfc = MolStandardize.fragment.LargestFragmentChooser()
        # standardize
        mol = Chem.MolFromSmiles(smi)
        mol2 = lfc.choose(mol)
        smi2 = Chem.MolToSmiles(mol2)
        #     print(smi2)
        #     # canonical
        #     can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi2))
        # #     print(can_smi)
        #     print(can_smi == smi2)
        return smi2

    def get_corpus_line(self, item):
        '''
            item : 返回
        '''
        # 返回相同的句子
        if self.on_memory:
            # 若果全部加入到内存，直接索引即可
            # 是一条smiles记录
            smiles = self.lines[item]
        else:
            # 否之逐行访问
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding="utf-8")
            curId = -1
            while curId < item:
                curId += 1
                line = self.file__next__()
            smiles = line
        if smiles[-1] == "\n":
            smiles = smiles[:-1]
        if smiles[-1] == " ":
            smiles = smiles[:-1]
        smiles = self.standardizeAndcanonical(smiles)
        # t = Chem.MolFromSmiles(smiles)
        # sentence_0 = mol2alt_sentence(t, 0)  # 获取半径为1的原子id
        # sentence = mol2alt_sentence(t, 1)
        # t = Chem.MolFromSmiles(smiles)
        sentence_1 = mol2alt_sentence(t, 1)
        # sentence_0 = mol2alt_sentence(t, 0)
        sentence = []
        for i in range(len(sentence_1)):
            if i % 2 != 0:
                sentence.append(sentence_1[i])
        # sentence = sentence_1
        # print(sentence,sentence_0)
        # label = int(line[1][:1])
        return sentence_1
        # return sentence[len(sentence_0):]


    # def get_random_line(self):
    #     if self.on_memory:
    #         return self.lines[random.randrange(len(self.lines))]["text2"]
    #
    #     line = self.random_file.__next__()
    #     if line is None:
    #         self.random_file.close()
    #         self.random_file = open(self.corpus_path, "r", encoding="utf-8")
    #         for _ in range(np.random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
    #             self.random_file.__next__()
    #         line = self.random_file.__next__()
    #     return eval(line)["text2"]
