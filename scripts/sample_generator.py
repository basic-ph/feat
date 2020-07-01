import csv
import logging
import time

import numpy as np

from feat import ge_wang, mesh


def main():

    Vf = 0.60
    side = 50
    w = 0.2  # (-11.5 * Vf**2 - 4.3*Vf + 8.5)  # empirical function
    
    seed = 11  # 96, 11, 50, 46, 88, 66, 89, 15, 33, 49
    rand_gen = np.random.default_rng(seed)
    radius = 1.0
    vertex = [0.0, 0.0, 0.0]
    min_dist_square = (2.1 * radius)**2
    max_iter = 100000
    
    centers = ge_wang.generate_rve(rand_gen, w, Vf, radius, vertex, side, min_dist_square, max_iter)
    logger.info("Algorithm generated fibers: %s", len(centers))
    centers = mesh.filter_centers(centers, radius, vertex, side)
    logger.info("Filtered fibers: %s", len(centers))
    logging.debug("Centers: \n%s", centers)

    filename = f"sample-{Vf}-{side}-{seed}.csv"
    data_file = f"../data/rve_samples/{filename}"
    with open(data_file, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(centers)
    logger.info("Output written to: %s", data_file)


if __name__ == "__main__":
    # LOGGING (you can skip this)
    log_lvl = logging.INFO
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
