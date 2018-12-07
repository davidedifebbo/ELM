import numpy as np
feature_maps = np.arange(0, 300).reshape(3, 10, 10)
pool_size = 2
new_feature_maps = np.zeros((feature_maps.shape[0], int(feature_maps.shape[1] / pool_size),
                             int(feature_maps.shape[2] / pool_size)))
for k in range(0, feature_maps.shape[0]):
    for i in range(0, int(feature_maps.shape[1] / pool_size)):
        for j in range(0, int(feature_maps.shape[2] / pool_size)):
            new_feature_maps[k][i][j] = feature_maps[k, i * pool_size:i * pool_size + pool_size,
                                                     j * pool_size:j * pool_size + pool_size].max()
