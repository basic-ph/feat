import numpy as np
from scipy import sparse

indptr = np.array([0, 2, 4, 7])  # length is n_col + 1, last item = number of values = length of both indices and data
indices = np.array([0, 2, 2, 2, 0, 1, 2])
data = np.array([1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0])  # nonzero values
a = sparse.csc_matrix((data, indices, indptr), shape=(3, 3))
print(a.toarray())
print((a.indptr))
print(a.indices)
print(a.data)
a.sum_duplicates()
print()
print(a.toarray())
print((a.indptr))
print(a.indices)
print(a.data)

# nel mio caso i valori non nulli sono tanti quanti i valori locali calcolati
# questo perché le K locali non hanno nessun valore nullo, tutti i dof dipendono da quelli vicini
# quindi per 10 elementi avremo 36x10=360 entries tutte non nulle
# il problema è quindi identificare tutte le collisioni (?)
# in realtà non mi interessa perché dovrei usare il metodo sum_duplicates che significa
# che comunque la matrice inizialmente mantiene i valori duplicati e quindi la memoria
# necessaria ad allocarla deve esserci