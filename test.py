import numpy as np
feature_maps = np.random.rand(3, 10, 10)
feature_maps = np.ceil(feature_maps)
number_of_kernel = 1
size_kernel = 3
n_feature = feature_maps.shape[0]
new_feature_maps = np.zeros((number_of_kernel, feature_maps.shape[1] - size_kernel + 1,
                             feature_maps.shape[2] - size_kernel + 1))
for k in range(0, number_of_kernel):
    kernel = np.random.rand(n_feature, size_kernel, size_kernel)
    kernel = np.ceil(kernel)
    for i in range(0, feature_maps.shape[1] - size_kernel + 1):
        for j in range(0, feature_maps.shape[2] - size_kernel + 1):
            new_feature_maps[k][i][j] = np.sum(
                np.multiply(feature_maps[:, i:i + size_kernel, j:j + size_kernel], kernel))
print()
