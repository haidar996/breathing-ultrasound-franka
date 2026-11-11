import os
# one maths thread per process (must be set before numpy is imported): the matrices here are
# tiny, and extra threads only fight each other and make everything slower
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
