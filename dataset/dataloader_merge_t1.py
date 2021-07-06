import sys
sys.path.append("../")
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
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
''''
应用于合并
'''
# 数据预处理和增强部分
class CLSDataset(Dataset):
    def __init__(self, corpus_path, word2idx, max_seq_len,
                 dataset_name="HIV", data_regularization=False, model=None, keys = None):

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
        self.model = model
        self.keys = keys
        print(corpus_path)
        # 加载语料
        with open(corpus_path, "r", encoding="utf-8") as f:
            # 将数据集全部加载到内存
            self.lines = [line for line in tqdm.tqdm(f, desc="Loading Dataset")]
            # 打乱顺序
            self.lines = shuffle(self.lines)
            # 获取数据长度(条数)
            self.corpus_lines = len(self.lines)
            if self.dataset_name == "freesolv" or self.dataset_name == 'esol' or \
                self.dataset_name == 'lipophilicity':
                # 标准化
                self.label = [float(line.split(",")[1]) for line in self.lines]
                self.max = max(self.label)
                self.min = min(self.label)
                self.k = 10 / (self.max - self.min)
                # pass

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # 得到tokenize之后的文本和与之对应的分类
        text, text_0, label, fp = self.get_text_and_label(item) # 获得对应的文本和标签

        text_input = self.tokenize_char(text)

        # 添加#CLS#和#SEP#特殊token
        text_input = [self.cls_index] + text_input + [self.sep_index]
        # 如果序列的长度超过self.max_seq_len限定的长度, 则截断
        output = {"text_input": torch.tensor(text_input),
                  "text_0": torch.tensor([text_0],dtype=torch.float32),
                  "label": torch.tensor([label]),
                  "fp": torch.tensor([fp],dtype=torch.float32)}

        # print(output)
        return output

    def standardizeAndcanonical(self,smi):
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

    def get_text_and_label(self, item):
        # 获取文本和标记

        if self.dataset_name == "hiv" or self.dataset_name == "bbbp" \
                or self.dataset_name == 'bace' or self.dataset_name == 'estrogen-alpha'\
            or self.dataset_name == 'estrogen-alpha' or self.dataset_name == 'estrogen-beta'\
                or self.dataset_name == 'mesta-high' or self.dataset_name == 'mesta-low':
            line = self.lines[item].split(",")
            # print(line)
            smiles = line[0]
            # print(smiles)
            if smiles[-1] == "\n":
                smiles = smiles[:-1]
            smiles = self.standardizeAndcanonical(smiles)
            t = Chem.MolFromSmiles(smiles)
            sentence_1 = mol2alt_sentence(t, 1)
            sentence_0 = mol2alt_sentence(t, 0)
            sentence = []
            for i in range(len(sentence_1)):
                if i % 2 != 0:
                    sentence.append(sentence_1[i])
            # sentence = sentence_1
            # print(sentence,sentence_0)
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
            sentence_1 = mol2alt_sentence(t, 1)
            sentence_0 = mol2alt_sentence(t, 0)
            sentence = []
            for i in range(len(sentence_1)):
                if i % 2 != 0:
                    sentence.append(sentence_1[i])
            label = float(line[1])
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
            sentence_1 = mol2alt_sentence(t, 1)
            sentence_0 = mol2alt_sentence(t, 0)
            sentence = []
            for i in range(len(sentence_1)):
                if i % 2 != 0:
                    sentence.append(sentence_1[i])
        # mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(t, 2, nBits=300)
        fp = np.array(fp)
        text_0 = self.sentences2vec(sentence_0)
        return sentence_1, text_0, label, fp

    def tokenize_char(self, segments):
        return [self.word2idx.get(char, self.unk_index) for char in segments]

    def sentences2vec(self, sentences, unseen=None):
        """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
        sum of vectors for individual words.

        Parameters
        ----------
        sentences : list, array
            List with sentences
        model : word2vec.Word2Vec
            Gensim word2vec model
        unseen : None, str
            Keyword for unseen words. If None, those words are skipped.
            https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

        Returns
        -------
        np.array
        """
        keys = self.keys
        # print(sentences)
        if unseen:
            unseen_vec = self.model.wv.word_vec(unseen)

        # if unseen:
        #     vec.append([self.model.wv.word_vec(y) if y in set(sentences) & keys
        #                     else unseen_vec for y in sentences])
        # else:
        #     vec.append([self.model.wv.word_vec(y) for y in sentences
        #                     if y in set(sentences) & keys])
        vec = np.array([0 for _ in range(300)])
        for y in sentences:
            if len(vec) == 0:
                vec = np.array(self.model.wv.word_vec(y))
            elif y in self.keys:
                vec = vec + np.array(self.model.wv.word_vec(y))
        # print(len(vec))
        return vec
