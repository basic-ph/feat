import argparse
import csv
import logging
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
import time

import numpy as np

import fem
import mesh


def main():

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
    realizations = 1  # number of realizations to compare
    Vf = 0.30  # fiber volume fraction
    radius = 1.0  # fiber radius
    number = 10  # number of fibers
    side = math.sqrt(math.pi * radius**2 * number / Vf)  # side lenght of the square domain (RVE)
    min_distance = 2.1 * radius
    offset = 1.1 * radius
    max_iter = 100000
    # mesh data
    # coarse_cls = [1.0, 0.5, 0.25, 0.12, 0.06]  # coarse element dimension (far from matrix-fiber boundary)
    coarse_cls = [0.25, 0.12, 0.06]  # coarse element dimension (far from matrix-fiber boundary)
    fine_cls = [cl / 2 for cl in coarse_cls]  # fine element dimension (matrix-fiber boundary)
    # fem data
    element_type = "triangle"

    
    i = 0
    max_i = 7
    storage = []
    x_centers = []  # array with x coordinate of centers
    y_centers = []
    old_side = None    

    while i < max_i:
        logger.info("------------------------------------------------------------")
        logger.info("Analysis of RVE #%s of size %s containing %s fibers", i+1, side, number)
        refined_moduli = []
        for r in range(realizations):  # loop over different realizations
            logger.debug("Analysis of realization #%s", r+1)
            # obtaining centers coordinates using RSA algorithm
            rand_gen = np.random.default_rng(7)  # random generator, accept seed as arg (reproducibility)
            x_centers, y_centers = mesh.get_fiber_centers(rand_gen, radius, number, side, min_distance, offset, max_iter, x_centers, y_centers, old_side)
            logger.debug("First fiber center obtained is: (%s, %s)", x_centers[0], y_centers[0])
            logger.debug("Fiber centers lenght: %s", len(x_centers))
            moduli = []

            for s in range(len(coarse_cls)):
                filename = f"validation-{i}-{r}-{s}"
                geo_path = "../data/geo/" + filename + ".geo"
                msh_path = "../data/reboot/" + filename + ".msh"
                coarse_cl = coarse_cls[s]  # 
                fine_cl = fine_cls[s]
                mesh_obj = mesh.create_mesh(
                    geo_path,
                    msh_path,
                    radius,
                    number,
                    side,
                    x_centers,
                    y_centers,
                    coarse_cl,
                    fine_cl
                )
                num_nodes = mesh_obj.points.shape[0]
                logger.debug("Created mesh details: coarse cl: %s | fine cl: %s | nodes: %s", coarse_cl, fine_cl, num_nodes)            
                # run FEM simulation on the current realization and mesh                
                E2 = analysis(mesh_obj, element_type, post_process=True, vtk_filename=filename)  # !!! works only with vector
                storage.append([i, r, s, side, num_nodes, E2])
                if s == 0:
                    moduli.append(E2) # store the value obtained for mesh convergence validation
                else:
                    moduli.append(E2)
                    prev_E2 = moduli[s-1] 
                    rel_diff = abs(E2 - prev_E2) / prev_E2  # difference relative to precedent obtained estimate
                    if rel_diff < 0.001:
                        logger.debug("Mesh convergence obtained for simulation #%s for realization #%s", s+1, r+1)
                        logger.debug("------------------------------")
                        refined_moduli.append(E2)  # saving the last values as the valid one
                        break  # mesh convergence obtained, continue with the next random realization

        mean_E2 = mean(refined_moduli)
        storage = [item + [mean_E2] if item[0] == i else item for item in storage]

        old_side = side  # saving the present side lenght for the next domain 
        i += 1
        number += 10
        side = math.sqrt(math.pi * radius**2 * number / Vf)  # ...causing the size of the RVE to increase
    
    logger.debug("Stored data:\n%s", storage)

    # date = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    data_file = f"../data/csv/reboot2.csv"
    with open(data_file, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(storage)
    logger.debug("Output written to: %s", data_file)

if __name__ == "__main__":
    # LOGGING (you can skip this)
    log_lvl = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(log_lvl)
    handler = logging.StreamHandler()
    handler.setLevel(log_lvl)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds ---")
