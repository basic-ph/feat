import time

import numpy as np

start_time = time.time()

# a = np.zeros(10000000)
# for i in range(10000000):
#     a[i] = 1

b = []
for i in range(10000000):
    b.append(1)

print("--- %s seconds ---" % (time.time() - start_time))    
