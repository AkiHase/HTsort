import os

######################################################
# dataset with '.h5' format in './h5data/h5filename'
######################################################
h5filename = "recordings_6cells_tetrode_30.0_10.0uV.h5"
pwd = os.getcwd()
h5data_path = os.path.join(pwd, "h5data", h5filename)

######################################################
# PARAMS
######################################################
threshold_coeff = 7   # ★★★ [6, (7), 8, 9]
# --------------spike detection range------
detect_ms_before = 0.75  #
detect_ms_after = 0.75   #
# --------------snippet range--------------
snippet_ms_before = 0.75  #
snippet_ms_after = 0.75
dect_frame_bias = 10  #
n_multiprocess = 4  # Number of cpu cores used

# ----------KNN outlier detection ---------
k_neighbor = 5
threshold_factor = 1.5  # [1.0, (1.5), 2.0]

# --------------decomposition--------------
n_components = 3
group_min_samples = 5  # ★★★ [5, 10, 20, (30)]

# ---------------HDBSCAN------------------
# min_cluster_size = 20
min_cluster_size = 5
# min_samples = None
min_samples = 1
outlier_recall_sigma = 0.05  # ★★★ [(0.3), 0.5, 1.0, ]

# -------------templates merging--------------
sigma = 0.05  # ★★★ sqidff [0.1, (0.05), 0.01]
merge_range = 2  #
peek_diff = 2.0  # not used

# -------------------templates matching--------
detect_th = threshold_coeff
# detect_th = 12
tm_coeff_bias = 0.02
peek_diff_tm = 5.0

if __name__ == "__main__":
    print("----------globvar.py------------")
    print(pwd)
    print(h5data_path)