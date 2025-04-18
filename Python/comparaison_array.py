import numpy as np


def pipeline_comparaison(arr1, arr2):
    all_close = np.allclose(arr1, arr2)
    print("np_allclose=", all_close)

    diff = np.abs(arr1 - arr2)
    print("Mean diff:", np.mean(diff))
    print("Max diff:", np.max(diff))

    diff_mask = ~np.isclose(arr1, arr2, rtol=1e-4, atol=1e-6)
    print("Pixels with difference >", np.sum(diff_mask))
