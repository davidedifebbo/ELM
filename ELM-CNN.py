import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import importlib
from sklearn.model_selection import train_test_split


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def ReLU(x, input_weights):
    a = np.dot(x, input_weights)
    a = np.maximum(a, 0, a)
    return a


def predict(x, input_weights, output_weights):
    x = ReLU(x, input_weights)
    y = np.dot(x, output_weights)
    return y


def load_CIFAR10_dataset():
    # Reading the CIFAR_10
    batch_1 = unpickle('data_batch_1')
    batch_2 = unpickle('data_batch_2')
    batch_3 = unpickle('data_batch_3')
    batch_4 = unpickle('data_batch_4')
    batch_5 = unpickle('data_batch_5')
    test_batch = unpickle('test_batch')

    # Concatenate all arrays
    train_data = np.concatenate((batch_1[b'data'], batch_2[b'data'], batch_3[b'data'],
                                 batch_4[b'data'], batch_5[b'data']))
    train_labels = np.concatenate((batch_1[b'labels'], batch_2[b'labels'], batch_3[b'labels'],
                                   batch_4[b'labels'], batch_5[b'labels']))
    test_data = np.array(test_batch[b'data'])
    test_labels = np.array(test_batch[b'labels'])
    return train_data, train_labels, test_data, test_labels


def vectorize_class_labels(labels_array, number_of_classes):
    labels_matrix = np.zeros([labels_array.shape[0], number_of_classes])
    for j in range(labels_array.shape[0]):
        labels_matrix[j][labels_array[j]] = 1
    return labels_matrix


def train_network(input_samples, output_targets, hidden_neurons, lambda_constant=0):
    input_weights = np.random.normal(size=[input_samples.shape[1], hidden_neurons])
    activations = ReLU(X_train, input_weights)
    activations_t = np.transpose(activations)
    output_weights = np.dot(np.linalg.inv(np.dot(activations_t, activations)), np.dot(activations_t, output_targets))
    return input_weights, output_weights, activations


def compute_accuracy(actual_output, target_output):
    correct = 0
    total = actual_output.shape[0]
    for i in range(total):
        predicted = np.argmax(actual_output[i])
        test = np.argmax(target_output[i])
        correct = correct + (1 if predicted == test else 0)
    return (correct / total) * 100


def conv3Dlayer(feature_maps, number_of_kernel, size_kernel):
    """
    :param feature_maps: 3D matrix containing all the feature maps from a layer
    :param number_of_kernel: number of filter the user wants to use on the feature maps
    :param size_kernel: number of elements of the kernel square matrix
    :return: returns the feature maps of the layer after
    """
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
    return new_feature_maps


########################################################################################################################

X_train, y_train, X_test, y_test = load_CIFAR10_dataset()

# Plot one image from CIFAR_10
rgb = np.reshape(X_train[0], (3, 32, 32))
rgb_rolled = np.rollaxis(rgb, 0, 3)
plt.imshow(rgb_rolled)

# normalizing features on a distribution with mean 0.0 and variance 1.0
X = X_train
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# Defining parameters network
n_features = X_train.shape[1]
n_hidden_neurons = 1000
n_classes = 10

# Transforming the labels into vector of '1' on the index value corresponding to the class and '0' otherwise
Y_train = vectorize_class_labels(y_train, n_classes)
Y_test = vectorize_class_labels(y_test, n_classes)

# Training the ELM
Win, Wout, H = train_network(X_train, Y_train, n_hidden_neurons)

Out = predict(X_test, Win, Wout)
accuracy = compute_accuracy(Out, Y_test)
print("Accuracy: ", accuracy, "%")

########################################################################################################################
