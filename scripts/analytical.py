from feat import micro


Ef = 100
Em = 10
Vf = 0.30
reuss_E2 = micro.reuss_model(Ef, Em, Vf)
print(reuss_E2)

halpin_tsai_E2 = micro.halpin_tsai_model(Ef, Em, Vf)
print(halpin_tsai_E2)