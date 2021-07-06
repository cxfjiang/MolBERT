import pickle
ident = open('ident_merge.pickle', 'rb')
tt = pickle.load(ident)
print(tt)
# print(tt)
idSize = len(tt)
print(idSize)
