import numpy as np

import contour_utils

def smooth(contours, scale):
    """
    Scale, means smoothing the contour by convolve the original contour with a gaussian kernel

    :param contours:
    :param scale_factor:
    :return:
    """
    n_point = len(contours)
    radius = n_point // 2

    weights = contour_utils.gaussian_kernel1d(sigma=scale, radius=radius)
    return contour_utils.convolve(contours, weights, radius=radius)


def group(plot_points, curvature):
    """

    :param plot_points:
    :param curvature:
    :return:
    """

    """
    Global parameters
    """
    neighbor_distance = 3.1
    min_point_per_cluster = 5

    plot_points = np.array(plot_points)
    plot_arclens = plot_points[:, 0]
    plot_scales = plot_points[:, 1]
    max_scale = plot_scales.max()
    n_point = curvature.shape[0]

    """
    Clustering
    """
    start_scale = max_scale
    clusters = []
    while (start_scale >= 3):
        ids = np.where(plot_scales == start_scale)[0]

        if len(clusters) == 0:
            # add new clusters
            for point_id in plot_arclens[ids]:
                clusters.append([(start_scale, point_id)])
        else:
            for point_id in plot_arclens[ids]:
                predicted_cluster_ids = []
                predicted_cluster_distances = []

                # calculate distance to each existing clusters
                for cluster_id, cluster_points in enumerate(clusters):
                    previous_scale = start_scale + 1

                    its_point_ids = [p[1] for p in cluster_points if p[0] == previous_scale]
                    its_distances = [abs(point_id - its_point_id) for its_point_id in its_point_ids]

                    predicted_cluster_id = -1
                    predicted_cluster_distance = np.inf
                    if its_distances and np.min(its_distances) < neighbor_distance:
                        predicted_cluster_id = cluster_id
                        predicted_cluster_distance = np.min(its_distances)

                    predicted_cluster_ids += [predicted_cluster_id]
                    predicted_cluster_distances += [predicted_cluster_distance]
                predicted_cluster_ids = np.array(predicted_cluster_ids)
                predicted_cluster_distances = np.array(predicted_cluster_distances)

                valid_cluster_ids = np.where(predicted_cluster_ids != -1)[0]
                if len(valid_cluster_ids) == 0:
                    if start_scale > 5: # small contour will be noisy
                        # new cluster
                        clusters.append([(start_scale, point_id)])

                elif len(valid_cluster_ids) >= 1:
                    if len(valid_cluster_ids) > 1:
                        # conflict -> no add
                        cluster_id = valid_cluster_ids[
                            np.argmin(predicted_cluster_distances[valid_cluster_ids])
                        ]

                    else:
                        # add to the cluster
                        cluster_id = predicted_cluster_ids[valid_cluster_ids[0]]

                    clusters[cluster_id] += [(start_scale, point_id)]

        start_scale -= 1

    """
    Localizing
    """
    localizing_points = []
    for cluster in clusters:
        # filter
        cluster = np.array(cluster)
        if len(cluster) < min_point_per_cluster: continue

        # the choosing peak is the one having the largest scale
        arclens = cluster[:, 1]
        scales = cluster[:, 0]
        max_id = np.argmax(scales)
        arclen, scale = arclens[max_id], scales[max_id]  # cluster[0]

        #
        cur_scale = scale
        while(cur_scale > 0):
            min_arclen = np.max([arclen - int(neighbor_distance), 0])
            max_arclen = np.min([arclen + int(neighbor_distance), n_point - 1])

            arclen = int(np.argmax(
                np.abs(curvature[min_arclen:(max_arclen + 1), cur_scale])
            )) + min_arclen

            cur_scale -= 1

        localizing_points.append([(arclen, cur_scale)])


    return localizing_points

