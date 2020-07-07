from feat import micro

Ef = 74
Em = 3.35
Vf = 0.60
reuss_E2 = micro.reuss_model(Ef, Em, Vf)
print(reuss_E2)

halpin_tsai_E2 = micro.halpin_tsai_model(Ef, Em, Vf)
print(halpin_tsai_E2)

"""
GE, WANG
reuss 7.142857142857142
H-T 8.829268292682928

Kaddour
reuss 7.8424549193293265
H-T 14.470321064996082
"""