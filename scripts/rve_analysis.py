import argparse
import csv
import logging
import math
from datetime import datetime
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
    rve_log.setLevel(rve_log_lvl)
    handler = logging.StreamHandler()
    handler.setLevel(rve_log_lvl)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # main_log formatter
    handler.setFormatter(formatter)
    rve_log.addHandler(handler)

    fem_log = logging.getLogger("fem")
    fem_log.setLevel(fem_log_lvl)
    fem_handler = logging.StreamHandler()
    fem_handler.setLevel(fem_log_lvl)
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
        prog="python rve_analysis.py",
        description=desc,
    )
    parser.add_argument(
        "-v",
        "-vector",
        help="trigger the usage of vectorized procedure",
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
    number = 10  # number of fibers
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

    
    i = 0
    max_i = 10
    storage = []

    while i < max_i:
        rve_log.info("----------------------------------------------------------------")
        rve_log.info("Analysis of RVE #%s of size %s containing %s fibers", i+1, side, number)
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
                mesh_obj = mesh.create_mesh(
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
                E2 = analysis(mesh_obj, element_type)
                storage.append([i, r, s, side, nodes, E2])
                if s == 0:
                    moduli.append(E2) # store the value obtained for mesh convergence validation
                else:
                    moduli.append(E2)
                    prev_E2 = moduli[s-1] 
                    rel_diff = abs(E2 - prev_E2) / prev_E2  # FIXME difference relative to precedent obtained estimate
                    if rel_diff < 0.01:
                        rve_log.info("Mesh convergence obtained for simulation #%s for realization #%s", s+1, r+1)
                        refined_moduli.append(E2)  # saving the last values as the valid one
                        break  # mesh convergence obtained, continue with the next random realization

        mean_E2 = mean(refined_moduli)
        storage = [item + [mean_E2] if item[0] == i else item for item in storage]
        threshold = 0.01 * mean_E2  # 5% of the mean
        upper_lim = mean_E2 + threshold
        lower_lim = mean_E2 - threshold

        max_E2 = max(refined_moduli)
        min_E2 = min(refined_moduli)

        if (min_E2 > lower_lim) and (max_E2 < upper_lim):
            rve_log.info("RVE #%s of size %s containing %s fibers has been validated!", i+1, side, number)
            rve_log.info("Mean transverse modulus is E2 = %s", mean_E2)
            break
        else:
            rve_log.info("RVE #%s of size %s is NOT representative!", i+1, side)
            i += 1
            number += 10
            side = math.sqrt(math.pi * radius**2 * number / Vf)  # ...causing the size of the RVE to increase
    
    rve_log.info("Stored data:\n%s", storage)

    # date = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    data_file = f"./data/output/rve_d.csv"
    with open(data_file, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(storage)

if __name__ == "__main__":
    main()
