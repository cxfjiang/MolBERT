import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import os
from rdkit.Chem import MolStandardize
import os
'''
   在zinc15上建立词典 
'''
def mol2alt_sentence(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]

    #     print(mol_atoms)
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)

allowedAtomsDict = {
    'H' : 1,'h' : 0,
    'B' : 5,'b' : 0,
    'C' : 6,'c' : 0,
    'N' : 7,'n' : 0,
    'O' : 8,'o' : 0,
    'F' : 9,'f' : 0,
    'P' : 15,'p': 0,
    'S' : 16,'s': 0,
    'Cl': 17,'Br' : 35
}
word = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZzBrCl"

def isValidCharacter(c):
    if c not in word or (c in word and c in "HhBbCcNnOoFfPpSsClBr"):
        return True
    return False

def isValidSmiles(smiles,atom_weight = 600,heavy_atom_count = 50):
    '''
        1. smiles能够被rdkit包处理
        2. smiles只包含特定元素
        3. smiles原子权重
    '''
    t_weight = 0
    heavyAtomCount = 0
    left = -len(smiles)-1
    right = -1
    idx = -1
    while True:
        if idx <= left:
            break
        c = smiles[idx]
        if smiles[idx] == 'r' or smiles[idx] == 'l' :
            c = (smiles[idx-1] if idx -1 > right else "#") + c
            idx = idx - 1
        idx = idx - 1
        if isValidCharacter(c) == True:
            if c in allowedAtomsDict.keys():
                t_weight = t_weight + int(allowedAtomsDict[c])
                heavyAtomCount = heavyAtomCount + (1 if int(allowedAtomsDict[c]) > 1 else 0)
        else:
            return False
#     print(type(t_weight),ttype(heavy_atom_count))
    return  True if t_weight >= 3 and t_weight <= atom_weight and heavyAtomCount <= heavy_atom_count else False

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

def main():
    ident = open('ident_merge.pickle', 'rb')
    tt = pickle.load(ident)
    idSize = len(tt)
    print(idSize)

    filePath = '/home/linxuan/my_nlp/Mol_BERT/dataset/zinc15/'
    total = 0
    validSmiles = 0
    t3 = 0

    for i, j, k in os.walk(filePath):
        for name in k:
            # 确定所有的文件名
            if '-' not in name and 'nohup.out' not in name:
                directory, fileName = name[:2], name
                with open(filePath + directory + "/" + fileName) as f:
                    # print(filePath + directory + "/" + fileName)
                    uu = 0
                    for smiles in f.readlines():
                        if uu == 0:
                            uu = 2
                            continue
                        smiles = smiles.split(" ")[0]
                        if smiles[-1] == "\n":
                            smiles = smiles[:-1]
                        smi = standardizeAndcanonical(smiles)
                        if isValidSmiles(smi) == True:
                            t3 += 1
                            t = Chem.MolFromSmiles(smi)
                            if t != None:  # 能够处理
                                sentence_rid_0 = mol2alt_sentence(t, 0)
                                sentence_rid_1 = mol2alt_sentence(t, 1)
                                if 2 * len(sentence_rid_0) == len(sentence_rid_1):
                                    for i in range(0, len(sentence_rid_0)):
                                        new_ident = sentence_rid_1[len(sentence_rid_0) + i] + sentence_rid_0[i]
                                        if new_ident not in tt.keys():
                                            tt[new_ident] = idSize
                                            idSize += 1
                                    validSmiles += 1
                                else:
                                    total += 1
            print(name, total, validSmiles)
    print(total, validSmiles, total, idSize)

    ## 持久化
    identification_file = open('ident_merge_v2.pickle', 'wb')
    pickle.dump(tt, identification_file)
    identification_file.close()

if __name__ == '__main__':
    main()

