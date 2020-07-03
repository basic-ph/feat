from feat import micro

Ef = 15
Em = 2.9
Vf = 0.60
reuss_E2 = micro.reuss_model(Ef, Em, Vf)
print(reuss_E2)

halpin_tsai_E2 = micro.halpin_tsai_model(Ef, Em, Vf)
print(halpin_tsai_E2)