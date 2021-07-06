import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import seaborn as sns
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
import pandas as pd
from help import Help
import pickle
import os

def mol2alt_sentence_new(mol, radius):
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

    # 初始化部分
    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    #     print(mol_atoms)
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}
    #     print(dict_atoms)

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}
    #     print(dict_atoms)
    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in [radius]:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt])  # not ignore the null indentification
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

def main():
    ident = open('ident.pickle', 'rb')
    tt = pickle.load(ident)
    # print(tt)
    idSize = len(tt)
    print(idSize)

    filePath = '../dataset/zinc15/'
    total = 0
    unGenerateMol = 0
    validSmiles = 0

    for i, j, k in os.walk(filePath):
        for name in k:
            # 确定所有的文件名
            if '-' not in name and 'nohup.out' not in name:
                directory, fileName = name[:2], name
                with open(filePath + directory + "/" + fileName) as f:
                    for smiles in f.readlines():
                        smiles = smiles.split(" ")[0]
                        if smiles[-1] == "\n":
                            smiles = smiles[:-1]
                        # print(smiles)
                        if isValidSmiles(smiles) == True:
                            t = Chem.MolFromSmiles(smiles)
                            if t == None:
                                unGenerateMol += 1
                            else:
                                sentence = mol2alt_sentence_new(t, 1)
                                if sentence[0] == 'None':
                                    unGenerateMol += 1
                                else:
                                    for sen in sentence:
                                        if sen not in tt:
                                            tt[sen] = idSize
                                            idSize += 1
                                    validSmiles += 1
                        total += 1
            print(name)  
    print('totalSize = ', idSize, "unGenerateMol = ", unGenerateMol, "validSmiles = ", validSmiles, "total = ", total)

    ## 持久化
    identification_file = open('ident_v2.pickle', 'wb')
    pickle.dump(tt, identification_file)
    identification_file.close()

if __name__ == '__main__':
    main()

