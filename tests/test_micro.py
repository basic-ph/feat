from feat import micro


def test_reuss_model():
    Ef = 224.4
    Em = 3.1
    Vf = 0.66
    output = micro.reuss_model(Ef, Em, Vf)
    assert round(output,2) == 8.88


def test_halpin_tsai_model():
    # DOI: 10.1080/15376494.2014.938792 (page 11)
    Ef = 74
    Em = 3.35
    Vf = 0.3

    output_1 = micro.halpin_tsai_model(Ef, Em, Vf)
    print(output_1)
    output_2 = micro.halpin_tsai_model(Ef, Em, Vf, xi=1)
    print(output_2)

    assert round(output_1, 3) == 6.930
    assert round(output_2, 3) == 5.879
