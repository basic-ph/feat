import math


def reuss_model(Ef, Em, Vf):
    # TODO check 0 < Vf < 1
    Vm = 1 - Vf
    E2 = (Ef * Em) / (Vm*Ef + Vf*Em)
    return E2


def halpin_tsai_model(Ef, Em, Vf, xi=2):
    # TODO check 0 < Vf < 1
    eta = ((Ef/Em)-1) / ((Ef/Em)+xi)
    E2 = Em * (1 + xi*eta*Vf)/(1 - eta*Vf)
    return E2