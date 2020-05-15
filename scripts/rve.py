import argparse
import csv
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

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
        prog="python rve.py",
        description=desc,
    )
    parser.add_argument(
        "version",
        help="specify which version of the FEA code should be used",
        choices=["base", "sparse", "vector"],
        default="vector",
        nargs="?"
    )
    args = parser.parse_args()

    if args.version == "base":
        analysis = fem.base_analysis
    elif args.version == "sparse":
        analysis = fem.sp_base_analysis
    else:  # if no version arg is provided the default is used
        analysis = fem.vector_analysis

    # DATA
    Vf = 0.30  # fiber volume fraction
    radius = 1.0  # fiber radius
    min_distance = 2.1 * radius
    offset = 1.1 * radius
    max_iter = 100000
    coarse_cls = [0.25, 0.12, 0.06]  #  [1.0, 0.5, 0.25, 0.12, 0.06] coarse element dimension (far from matrix-fiber boundary)
    fine_cls = [cl / 2 for cl in coarse_cls]  # fine element dimension (matrix-fiber boundary)
    element_type = "triangle"

    data = []  # list of [s: id del sample, n: numero di fibre nel dominiio, coarse_cl, side, num_nodes, E2, {0|1}]
    s = 0

    seeds = [96, 11, 50, 46, 88, 53, 89, 15, 33, 49]
    samples = 3  # decides how many generations (runs) have to be completed
    max_n = 30  # maximus number of fibers, decides how many domains should be generated for each "sample"

    while s < samples:
        n = 10
        side = math.sqrt(math.pi * radius**2 * n / Vf)  # side lenght of the square domain (RVE)

        x_centers = []
        y_centers = []
        old_side = None
        rand_gen = np.random.default_rng(seeds[s])  # create the generation for all different domains
        logger.info("------------------------------------------------------------")
        logger.info("Analysis of sample #%s with random seed: %s", s+1, seeds[s])

        while n <= max_n: # loop over the different domain with increasing number of fibers
            logger.info("------------------------------------------------------------")
            logger.info("RVE with %s fibers", n)
            moduli = []  # clean list used for mesh convergence validation
            
            x_centers, y_centers = mesh.get_fiber_centers(rand_gen, radius, n, side, min_distance, offset, max_iter, x_centers, y_centers, old_side)
            logger.debug("First fiber center obtained is: (%s, %s)", x_centers[0], y_centers[0])
            logger.debug("Fiber centers lenght: %s", len(x_centers))
            
            for m in range(len(coarse_cls)):
                filename = f"rve-{s}-{n}-{m}"
                geo_path = "../data/geo/" + filename + ".geo"
                msh_path = "../data/msh/" + filename + ".msh"
                coarse_cl = coarse_cls[m]
                fine_cl = fine_cls[m]
                mesh_obj = mesh.create_mesh(
                    geo_path,
                    msh_path,
                    radius,
                    n,
                    side,
                    x_centers,
                    y_centers,
                    coarse_cl,
                    fine_cl
                )
                num_nodes = mesh_obj.points.shape[0]
                # run FEM simulation
                E2 = analysis(mesh_obj, element_type)

                if m == 0:  # first mesh
                    moduli.append(E2) # store the value obtained for mesh convergence validation
                    data.append([s, n, coarse_cl, side, num_nodes, E2, 0])  # 0 means non-converged result
                else:
                    moduli.append(E2)
                    prev_E2 = moduli[m-1]
                    rel_diff = abs(E2 - prev_E2) / prev_E2  # difference relative to precedent obtained estimate
                    if rel_diff < 0.001:
                        data.append([s, n, coarse_cl, side, num_nodes, E2, 1])  # 1 means converged result
                        logger.info("Mesh #%s converged!", m+1)
                        break  # mesh convergence obtained, continue with the next random realization
                    else:
                        data.append([s, n, coarse_cl, side, num_nodes, E2, 0])  # 0 means non-converged result
                        logger.info("Mesh #%s NOT converged!", m+1)
            old_side = side  # saving the present side lenght for the next domain
            n += 10
            side = math.sqrt(math.pi * radius**2 * n / Vf)  # ...causing the size of the RVE to increase

        s += 1


    actual_date = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    data_file = f"../data/csv/{actual_date}.csv"
    with open(data_file, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data)
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