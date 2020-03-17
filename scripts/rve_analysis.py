import logging
from statistics import stdev
from pathlib import Path


def main():
    # LOGGING
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()  # main_log handler
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # main_log formatter
    handler.setFormatter(formatter)
    log.addHandler(handler)

    # DATA
    realiz_num = 10  # number of realizations
    rve_size = 10  # side lenght of the square domain (RVE)
    V_f = 0.5  # fibers volume fraction

    moduli_list = []
    for r in range(realiz_num):  # loop over different realizations
        log.info("Analyzing realization #%s", r+1)

        # here goes the sequence of FEM analysis: mesh refinement
        # how is convergence of mesh verified mathematically?
        E2 = 25.0  # this is the return value from a complete FEM analysis
        moduli_list.append(E2)  # insert the value for each realization in the list

    if stdev(moduli_list) < 5:  # which is the right threshold??
        pass


if __name__ == "__main__":
    main()
