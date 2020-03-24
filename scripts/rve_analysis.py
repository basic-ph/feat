import argparse
import logging
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

import fem
import mesh


def main():
    # LOGGING (you can skip this)
    rve_log_lvl = logging.INFO
    fem_log_lvl = logging.INFO
    feat_log_lvl = logging.DEBUG

    rve_log = logging.getLogger("rve")
    rve_log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # main_log formatter
    handler.setFormatter(formatter)
    rve_log.addHandler(handler)

    fem_log = logging.getLogger("fem")
    fem_log.setLevel(logging.INFO)
    fem_handler = logging.StreamHandler()
    fem_handler.setLevel(logging.INFO)
    fem_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # fem_log formatter
    fem_handler.setFormatter(fem_formatter)
    fem_log.addHandler(fem_handler)

    feat_log = logging.getLogger("feat")
    feat_log.setLevel(feat_log_lvl)
    feat_handler = logging.StreamHandler()
    feat_handler.setLevel(feat_log_lvl)
    feat_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    feat_handler.setFormatter(feat_formatter)
    feat_log.addHandler(feat_handler)

    # argument parser creation and setup
    desc = (
        "Complete an RVE analysis for the evaluation of transverse modulus "
        "of unidirectional fiber-reinforced composite materials"
    )
    parser = argparse.ArgumentParser(
        prog="RVE Analysis",
        description=desc,
    )
    parser.add_argument(
        "-v",
        "-vector",
        help="flag that triggers the usage of vectorized procedure",
        dest="vector",
        action="store_true",
    )
    args = parser.parse_args()

    # check if flag arg is given and link "analysis" to the wanted simulation function
    if args.vector:
        analysis = fem.vector_analysis
    else:
        analysis = fem.base_analysis

    # DATA
    # rve data
    realizations = 10  # number of realizations to compare
    Vf = 0.30  # fiber volume fraction
    radius = 1.0  # fiber radius
    number = 0  # number of fibers
    side = math.sqrt(math.pi * radius**2 * number / Vf)  # side lenght of the square domain (RVE)
    min_distance = 2.1 * radius
    offset = 1.1 * radius
    max_iter = 100000
    # mesh data
    coarse_cls = [1.0, 0.5, 0.25, 0.12]  # coarse element dimension (far from matrix-fiber boundary)
    fine_cls = [cl / 2 for cl in coarse_cls]  # fine element dimension (matrix-fiber boundary)
    # fem data
    element_type = "triangle"

    # -------------- FROM MESH.PY --------------------
    # geo_path = "data/geo/rve_1.geo"  # TODO particular name for each realiz and mesh
    # msh_path = "data/msh/rve_1.msh"

    rve_log.info("--------------------------------------------------------------------")
    i = 0
    max_i = 20
    storage = []
    refined_storage = []

    while i < max_i:
        i += 1
        number += 5  # increase the number of fibers...
        side = math.sqrt(math.pi * radius**2 * number / Vf)  # ...causing the size of the RVE to increase
        rve_log.info("Analysis of RVE #%s of size %s", i, side)
        refined_moduli = []
        for r in range(realizations):  # loop over different realizations
            rve_log.info("Analysis of realization #%s", r+1)
            # obtaining centers coordinates using RSA algorithm
            x_array, y_array = mesh.get_fiber_centers(radius, number, side, min_distance, offset, max_iter)
            moduli = []

            for s in range(len(coarse_cls)):
                filename = f"rve-{i}-{r+1}-{s+1}"
                geo_path = "data/geo/" + filename + ".geo"
                msh_path = "data/msh/" + filename + ".msh"
                coarse_cl = coarse_cls[s]  # 
                fine_cl = fine_cls[s]
                mesh_obj = mesh.create_mesh(   # TODO maybe use the obj directly instead of reading from .msh
                    geo_path,
                    msh_path,
                    radius,
                    number,
                    side,
                    x_array,
                    y_array,
                    coarse_cl,
                    fine_cl
                )
                nodes = mesh_obj.points.shape[0]
                rve_log.info("Created mesh details: coarse cl: %s | fine cl: %s | nodes: %s", coarse_cl, fine_cl, nodes)            
                # run FEM simulation on the current realization and mesh                
                E2 = analysis(msh_path, element_type)
                if s == 0:
                    moduli.append(E2) # store the value obtained for mesh convergence validation
                else:
                    moduli.append(E2)
                    prev_E2 = moduli[s-1] 
                    rel_diff = abs(E2 - prev_E2) / prev_E2  # FIXME difference relative to precedent obtained estimate
                    if rel_diff < 0.05:
                        rve_log.info("Mesh convergence obtained for simulation #%s for realization #%s", s+1, r+1)
                        refined_moduli.append(E2)  # saving the last values as the valid one
                        storage.append([i, r, s, side, nodes, E2])
                        break  # mesh convergence obtained, continue with the next random realization

        mean_E2 = mean(refined_moduli)
        refined_storage.append([i, side, mean_E2])
        threshold = 0.05 * mean_E2  # 5% of the mean
        upper_lim = mean_E2 + threshold
        lower_lim = mean_E2 - threshold

        max_E2 = max(refined_moduli)
        min_E2 = min(refined_moduli)

        if (min_E2 > lower_lim) and (max_E2 < upper_lim):
            rve_log.info("RVE #%s of size %s validated!", i, side)
            rve_log.info("Mean transverse modulus is E2 = %s", mean_E2)
            break
        else:
            rve_log.info("RVE #%s of size %s is NOT representative!", i, side)
    
    rve_log.info("Stored data:\n%s", storage)
    rve_log.info("Stored refined data:\n%s", refined_storage)


if __name__ == "__main__":
    main()
