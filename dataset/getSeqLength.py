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

dataset_name = ['hiv','bbbp','bace',"tox21",
    "sider","clintox","esol","freesolv",'lipophilicity']

dataset_seqLength = {}

def standardizeAndcanonical(smi):
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


for datasetName in dataset_name:
    seq_len_less_50 = 0
    seq_len_less_100 = 0
    seq_len_less_150 = 0
    seq_len_less_200 = 0
    seq_len_large_200 = 0

    with open(datasetName + "/train_0.txt", "r") as f:
        for smlAndLabel in f.readlines():
            sml = smlAndLabel.split(",")[0]
            # print(sml)
            smiles = standardizeAndcanonical(sml)
            t = Chem.MolFromSmiles(smiles)
            sentence_0 = mol2alt_sentence(t, 0)

            length = len(sentence_0)
            if length <= 50:
                seq_len_less_50 += 1

            elif length <= 100:
                seq_len_less_50 += 1
                seq_len_less_100 += 1

            elif length <= 150:
                seq_len_less_50 += 1
                seq_len_less_100 += 1
                seq_len_less_150 += 1

            elif length <= 200:
                seq_len_less_50 += 1
                seq_len_less_100 += 1
                seq_len_less_150 += 1
                seq_len_less_200 += 1
            else:
                seq_len_large_200 += 1
    with open(datasetName + "/test_0.txt", "r") as f:
        for smlAndLabel in f.readlines():
            sml = smlAndLabel.split(",")[0]
            smiles = standardizeAndcanonical(sml)
            t = Chem.MolFromSmiles(smiles)
            sentence_0 = mol2alt_sentence(t, 0)
            length = len(sentence_0)
            if length <= 50:
                seq_len_less_50 += 1

            elif length <= 100:
                seq_len_less_50 += 1
                seq_len_less_100 += 1

            elif length <= 150:
                seq_len_less_50 += 1
                seq_len_less_100 += 1
                seq_len_less_150 += 1

            elif length <= 200:
                seq_len_less_50 += 1
                seq_len_less_100 += 1
                seq_len_less_150 += 1
                seq_len_less_200 += 1
            else:
                seq_len_large_200 += 1
    with open(datasetName + "/validation_0.txt", "r") as f:
        for smlAndLabel in f.readlines():
            sml = smlAndLabel.split(",")[0]
            smiles = standardizeAndcanonical(sml)
            t = Chem.MolFromSmiles(smiles)
            sentence_0 = mol2alt_sentence(t, 0)
            length = len(sentence_0)
            if length <= 50:
                seq_len_less_50 += 1

            elif length <= 100:
                seq_len_less_50 += 1
                seq_len_less_100 += 1

            elif length <= 150:
                seq_len_less_50 += 1
                seq_len_less_100 += 1
                seq_len_less_150 += 1

            elif length <= 200:
                seq_len_less_50 += 1
                seq_len_less_100 += 1
                seq_len_less_150 += 1
                seq_len_less_200 += 1
            else:
                seq_len_large_200 += 1
    dataset_seqLength[datasetName] = {"<=50":seq_len_less_50,
                              "<=100":seq_len_less_100,
                              "<=150":seq_len_less_150,
                            "<=200":seq_len_less_200,
                            ">200":seq_len_large_200}
print(dataset_seqLength)



