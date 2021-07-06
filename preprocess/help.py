class Help:

    def __init__(self):

        self.allowedAtomsDict = {
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

        self.word = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZzBrCl"
        self.allowedAtoms = "HhBbCcNnOoFfPpSsClBr"

    def isValidCharacter(self, c):
        if c not in self.word or (c in self.word and c in self.allowedAtoms):
            return True
        return False

    def isValidSmiles(self, smiles, atom_weight=600, heavy_atom_count=50):
        '''
            1. smiles能够被rdkit包处理
            2. smiles只包含特定元素
            3. smiles原子权重
        '''
        t_weight = 0
        heavyAtomCount = 0
        left = -len(smiles) - 1
        right = -1
        idx = -1
        while True:
            if idx <= left:
                break
            c = smiles[idx]
            if smiles[idx] == 'r' or smiles[idx] == 'l':
                c = (smiles[idx - 1] if idx - 1 > right else "#") + c
                idx = idx - 1
            idx = idx - 1
            if self.isValidCharacter(c):
                if c in self.allowedAtomsDict.keys():
                    t_weight = t_weight + int(self.allowedAtomsDict[c])
                    heavyAtomCount = heavyAtomCount + (1 if int(self.allowedAtomsDict[c]) > 1 else 0)
            else:
                return False
        #     print(type(t_weight),ttype(heavy_atom_count))
        return True if 3 <= t_weight <= atom_weight and heavyAtomCount <= heavy_atom_count else False