from copy import deepcopy
from numpy import argmin, array, mean, std, zeros
from numpy.linalg import norm
from numpy.random import random


def k_mean(data, cluster_count):
    # since data_size = data.shape[0],
    feature_dimension = data.shape[1]
    data_mean = mean(data)
    data_std = std(data)
    curr_centers = random((cluster_count, feature_dimension)) * data_std + data_mean
    prev_centers = zeros(curr_centers.shape)
    converged = False

    while not converged:
        # distance = ∑∑(x - ci)
        distances = array([norm(data - curr_centers[ci], axis=1) for ci in range(cluster_count)]).T
        # get indexes of center which individual data is closest to
        center_indexes_closest_to_data = argmin(distances, axis=1)
        for i in range(cluster_count):
            data_closest_to_this_center = data[center_indexes_closest_to_data == i]
            if len(data_closest_to_this_center) > 0:
                curr_centers[i] = mean(data_closest_to_this_center, axis=0)
        error = norm(curr_centers - prev_centers)
        converged = error == 0
        prev_centers = deepcopy(curr_centers)

    distances = array([norm(data - curr_centers[ci], axis=1) for ci in range(cluster_count)]).T
    center_indexes_closest_to_data = argmin(distances, axis=1)
    clusters = []
    standard_deviations = zeros(cluster_count)
    for i in range(cluster_count):
        data_closest_to_this_center = data[center_indexes_closest_to_data == i]
        clusters.append(data_closest_to_this_center)
        if len(data_closest_to_this_center) > 1:
            standard_deviations[i] = std(data_closest_to_this_center)
    return array(clusters), curr_centers, standard_deviations


# example:
if __name__ == "__main__":
    samples = array([
        [1, 2],
        [4, 5],
        [7, 3],
        [1, 4],
        [4, 2],
        [7, 1],
        [1, 1],
        [4, 4],
        [7, 2],
        [1, 3],
        [4, 3],
        [7, 5],
    ])
    clusters, centers, standard_deviations = k_mean(samples, 3)
    for i, cluster in enumerate(clusters):
        print("{}th center: {}".format(i, centers[i]))
        print("{}th std: {}".format(i, standard_deviations[i]))
        print("{}th cluster: {}".format(i, cluster))
