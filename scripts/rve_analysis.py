import logging
from statistics import stdev, mean
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

    max_iter = 10
    i = 0

    while i < max_iter:

        moduli_list = []
        for r in range(realiz_num):  # loop over different realizations
            log.info("Analyzing realization #%s", r+1)

            # here goes the sequence of FEM analysis: mesh refinement
            # how is convergence of mesh verified mathematically?
            E2 = 25.0  # this is the return value from a complete FEM analysis
            moduli_list.append(E2)  # insert the value for each realization in the list

        threshold = 0.05 * mean(moduli_list)  # 5% of the mean
        upper_lim = mean(moduli_list) + threshold
        lower_lim = mean(moduli_list) - threshold

        max_E2 = max(moduli_list)
        min_E2 = min(moduli_list)

        if (min_E2 > lower_lim) and (max_E2 < upper_lim):
            print("validated RVE")
            break
        else:
            print("RVE size is not enough...")


if __name__ == "__main__":
    main()
