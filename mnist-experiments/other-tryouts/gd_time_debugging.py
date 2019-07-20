import settings
import pickle

import exp_cluster_attr_test_GD
import exp_outlier_test_GD
import exp_letter_test_GD


#fname = exp_cluster_attr_test_GD.generate_time_results_filename(parameters=settings.parameters)
fname = exp_outlier_test_GD.generate_time_results_filename(parameters=settings.parameters)
#fname = exp_letter_test_GD.generate_time_results_filename(parameters=settings.parameters)


with open(fname, "rb") as f:
    a = pickle.load(f)

for j in range(170):
    print("================== ", j)
    print(a[0][j])
    print(a[1][j])
    print(a[2][j])
    print(a[3][j])
    print(a[4][j])
    print(a[5][j])
    print(a[6][j])
    print(a[7][j])
    print(a[8][j] if j<len(a[8]) else '-')

