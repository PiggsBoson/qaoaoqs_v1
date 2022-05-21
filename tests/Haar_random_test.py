import sys
sys.path.insert(1, '/Users/william/OneDrive - HKUST Connect/Codes/QAOA/qaoaoqs_v1/')
from result_analysis.result import *
from qaoaoqs.tools import *

res = Result(1,1,1)
# density_mtx = [res.naive_rd_state() for _ in range(2000)]
density_mtx = [res.Haar_rd_state(1) for _ in range(2000)]
plot_bloch_sphere(density_mtx)
