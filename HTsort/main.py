import numpy as np
import numpy as pd
from scipy.io import loadmat, savemat
import globvar
import copy
from extractor import *
from detection import Detection, decomposition
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter
import copy
import cv2
import time
import csv
from sklearn.decomposition import PCA
import hdbscan
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process, Pool
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw


def cluster(raw_data: np.ndarray, decomp_data: np.ndarray, min_cluster_size=globvar.min_cluster_size, min_samples=globvar.min_samples):
    '''
    Parameters
    -----------------------
    raw_data: np.ndarray
    decomp_data: np.ndarray
    min_cluster_size: int
        superparameter of DBSCAN

    Returns
    ----------------------
    labels: np.ndarray
    centroids: np.ndarray
        shape: [n_clusters, n_rawfeatures]
    '''
    if len(decomp_data) <= globvar.group_min_samples:
        return np.ones(len(decomp_data)).astype(np.int) * -1, np.array([]).reshape([0, raw_data.shape[1]])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, allow_single_cluster=True)
    clusterer.fit(decomp_data)
    labels: np.ndarray = clusterer.labels_

    n_cluters = np.max(labels) + 1
    centroids = np.zeros([n_cluters, raw_data.shape[1]])  # templates
    for i in range(n_cluters):
        thislabel_rawdata = raw_data[np.where(labels == i)]
        # centroids[i] = np.mean(thislabel_rawdata, axis=0)
        centroids[i] = np.median(thislabel_rawdata, axis=0)

    outlier_ids = np.where(labels == -1)[0]
    outlier_snips = raw_data[outlier_ids]
    for i in range(len(outlier_snips)):
        for j in range(n_cluters):
            sqdiff, loc = match_temp1d(outlier_snips[i], centroids[j], method_sel=5)
            if sqdiff <= globvar.outlier_recall_sigma:
                labels[outlier_ids[i]] = j

    return labels, centroids


def cluster_for_centroids(centroids: np.ndarray):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, allow_single_cluster=False)
    clusterer.fit(centroids)
    labels: np.ndarray = clusterer.labels_
    return labels

def tm_sqdiff(array: np.ndarray, tempalte: np.ndarray):
    sqdiff = np.sum(np.square(array - tempalte))
    return sqdiff


def plot_tempgroups_list(tempgroups_list: list):
    if len(tempgroups_list) <= 6:
        colors = ['r', 'g', 'b', 'm', 'y', 'c']
    else:
        cm_subsection = np.linspace(0, 1, len(tempgroups_list))
        colors = [cm.jet(x) for x in cm_subsection]

    plt.figure()
    for i in range(len(tempgroups_list)):
        if len(tempgroups_list[i]) == 0:
            continue
        plt.plot(tempgroups_list[i].T, c=colors[i], label=i)
    plt.legend()
    now_time = time.strftime("%Y-%m-%d %H,%M,%S", time.localtime())
    plt.savefig(f"centroids{now_time}.png")

    for i in range(len(tempgroups_list)):
        if len(tempgroups_list[i]) == 0:
            continue
        for j in range(i + 1, len(tempgroups_list)):
            if len(tempgroups_list[j]) == 0:
                continue
            for k in range(len(tempgroups_list[i])):
                temp_ik = tempgroups_list[i][k]
                temps_j = tempgroups_list[j]
                coeff_list = []
                sqdiff_list = []
                for b in range(len(temps_j)):
                    coeff, loc = match_temp1d(temp_ik, temps_j[b], method_sel=1)
                    sqdiff, loc2 = match_temp1d(temp_ik, temps_j[b], method_sel=5)
                    # sqdiff = tm_sqdiff(temp_ik, temps_j[b])
                    coeff_list.append(coeff)
                    sqdiff_list.append(sqdiff)
                print(f"[{i}, {k}]-[{j}]: coeff: {coeff_list}, sqdiff: {sqdiff_list}")

            # plt.figure()
            # if len(temps_j) <= 6:
            #     colors = ['r', 'g', 'b', 'm', 'y', 'c']
            # else:
            #     cm_subsection = np.linspace(0, 1, len(temps_j))
            #     colors = [cm.jet(x) for x in cm_subsection]
            # plt.plot(temp_ik, c='k')
            # for e in range(len(temps_j)):
            #     plt.plot(temps_j[e], c=colors[e])
            # plt.legend(labels=np.arange(len(temps_j + 1)))
            # plt.title(f"[{i}, {k}]-[{j}]: {coeff_list}")
            # plt.savefig(f"{i}-{j}-{k}.png")


def templates_merging(templategroups_list_: list, labelgroup_list_: list, channel_location: list, channel_type: str, sigma, peek_diff):
    '''
    Parameters
    --------------------
    templategroups_list_: list
        list of np.ndarray
    labelgroup_list_: list
        list of np.ndarray
    channel_location: list
        location of each channel
    channel_type: str
        such as "tetrode"  "Neuropixels"
    sigma: float
        The similarity between two templates has to be greater than that value to have a chance of merging
    peek_diff: float
    '''
    templategroups_list = copy.deepcopy(templategroups_list_)
    labelgroup_list = copy.deepcopy(labelgroup_list_)
    n_groups = len(templategroups_list)
    labels_map = []
    labels_map_flag = []
    n_features = templategroups_list[0].shape[1]

    # initial labels_map
    for k in range(n_groups):
        labels_map.append(np.arange(len(templategroups_list[k])))
        labels_map_flag.append(np.zeros(len(templategroups_list[k]), dtype=np.int))

    for i in range(n_groups):
        this_tempgroup = templategroups_list[i]
        if len(this_tempgroup) == 0:
            continue
        ch_loc_i = channel_location[i]
        labels_cnt = np.max(labels_map[i]) + 1
        for j in range(i + 1, n_groups):
            other_tempgroup = templategroups_list[j]
            if len(other_tempgroup) == 0:
                continue
            ch_loc_j = channel_location[j]
            loc_sqdistance = np.array([(ch_loc_i[0] - ch_loc_j[0]) ** 2 + (ch_loc_i[1] - ch_loc_j[1]) ** 2])
            loc_sqdistance_all = np.array([])
            for i_loc in range(len(channel_location)):
                if i_loc == i:
                    continue
                else:
                    loc_sqdistance_all = np.append(loc_sqdistance_all, (ch_loc_i[0] - channel_location[i_loc][0]) ** 2 + (ch_loc_i[1] - channel_location[i_loc][1]) ** 2)
            loc_sqdistance_all = np.unique(loc_sqdistance_all)
            # "tetrode"  "Neuropixels"
            can_merge_flag = 0  #
            if channel_type == "tetrode":
                if loc_sqdistance <= np.min(loc_sqdistance_all):
                # if (loc_sqdistance <= np.sort(loc_sqdistance_all)[: 2]).any():
                    can_merge_flag = 1
            else:
                if (loc_sqdistance <= np.sort(loc_sqdistance_all)[: globvar.merge_range]).any():
                    can_merge_flag = 1

            # print(f"loc_sqdistance: {loc_sqdistance}")
            # print(f"loc_sqdistance_all: {np.sort(loc_sqdistance_all)}")
            # print(f"can_merge_flag:{can_merge_flag}")

            mergeids_ingroupJ = np.array([], dtype=np.int)
            if can_merge_flag != 0:
                for i_temp in range(len(templategroups_list[i])):
                    this_temp = this_tempgroup[i_temp]
                    peek_diff_array = np.array([])
                    coeff_array = np.array([])
                    for j_temp in range(len(other_tempgroup)):
                        that_temp = other_tempgroup[j_temp]
                        coeff, loc = match_temp1d(this_temp, that_temp, method_sel=5)
                        # coeff = tm_sqdiff(this_temp, that_temp)
                        coeff_array = np.append(coeff_array, coeff)
                        peek_diff_array = np.append(peek_diff_array, np.abs(this_temp[n_features // 2] - that_temp[n_features // 2]))
                    # print(f"coeff_array:{coeff_array}, peek_diff_array:{peek_diff_array}, peek_diff{peek_diff}")
                    # can_merge_ids = np.copy(np.where((coeff_array <= sigma) & (peek_diff_array <= peek_diff))).flatten()  # can_merge_ids 正常来说就单个值，不会有多个

                    # print(f"g{i}-temp{i_temp} to g{j}, coeff_array{coeff_array}")

                    can_merge_ids = np.where(coeff_array <= sigma)[0]
                    # print(f"can_merge_ids:{can_merge_ids}")
                    mergeids_ingroupJ = np.append(mergeids_ingroupJ, can_merge_ids)
                    if len(can_merge_ids) != 0:
                        temps_tomerge = np.vstack((this_temp, other_tempgroup[can_merge_ids]))
                        weights = np.array([np.sum(labelgroup_list[i] == i_temp)], dtype=np.int)
                        for each_id in can_merge_ids:
                            weights = np.append(weights, np.sum(labelgroup_list[j] == each_id))
                        merged_temp = np.average(temps_tomerge, weights=weights, axis=0)
                        templategroups_list[i][i_temp] = merged_temp
                        # templategroups_list[j] = np.delete(templategroups_list[j], can_merge_ids, axis=0)  # 注意：不要即时删除，否则can_merge_ids已经无法正确对应到因为即时删除而更新的templategroups_list[j]
                        # update lables_map
                        labels_map[j][can_merge_ids] = labels_map[i][i_temp]
                        labels_map_flag[j][can_merge_ids] = 1
            templategroups_list[j] = np.delete(templategroups_list[j], mergeids_ingroupJ, axis=0)

            for n in range(len(labels_map[j])):
                if n not in mergeids_ingroupJ and labels_map_flag[j][n] != 1:
                    labels_map[j][n] = labels_cnt
                    labels_cnt += 1

            # print(f"mergeids_ingroupJ: {mergeids_ingroupJ}")
            # for f in range(len(labels_map)):
            #     print(f"g{f} {labels_map[f]}")

    for m in range(n_groups):
        ids_buff = []
        if len(labels_map[m]) == 0:
            continue
        for l in range(len(labels_map[m])):
            ids_buff.append(
                np.copy(np.where(labelgroup_list[m] == l)).flatten())
        for l in range(len(labels_map[m])):
            labelgroup_list[m][ids_buff[l]] = labels_map[m][l]

    merged_centroids = np.zeros([0, templategroups_list[0].shape[1]])
    for i in range(n_groups):
        if len(templategroups_list[i]) == 0:
            continue
        merged_centroids = np.vstack((merged_centroids, templategroups_list[i]))
    return merged_centroids, labelgroup_list


# KNN outlier detection (NOT USED)
def knn_outliner(data, K=None, threshold_factor=1.5):
    all_dist = []
    for i in range(data.shape[0]):
        diff_i = np.delete(data, i, axis=0) - data[i]
        # print("diff_i.shape: {}".format(diff_i.shape))
        dist_i = np.sum(np.square(diff_i), axis=1)
        if K is not None:
            avg_dist_i = np.mean(np.sort(dist_i)[:K])
        else:
            avg_dist_i = np.mean(np.sort(dist_i)[:])
        all_dist.append(avg_dist_i)

    q1 = np.quantile(all_dist, 0.25)
    q3 = np.quantile(all_dist, 0.75)
    iqr = q3 - q1
    threshold = q3 + threshold_factor * iqr
    # threshold = threshold_factor * q3
    is_outline = all_dist > threshold

    labels_triaged = np.copy(is_outline).astype(np.int16)
    labels_triaged[np.where(is_outline == 1)] = -1
    labels_triaged[np.where(is_outline == 0)] = 0

    return labels_triaged


def match_temp1d(data, temp, method_sel=1):
    """
    :param method_sel : match method
    :param data: target
    :param temp: template
    :return coeff: match result
    """
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
               'cv2.TM_SQDIFF_NORMED']
    data = data.reshape([1, -1])
    spike_data = np.array(np.copy(data), dtype=np.float32)
    temp = temp.reshape([1, -1])
    temp = np.array(np.copy(temp), dtype=np.float32)
    method_str = methods[method_sel]
    tm_method = eval(method_str)

    res = cv2.matchTemplate(spike_data, temp, tm_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method_str == methods[4] or method_str == methods[5]:
        # only for "TM_SQDIFF" method
        coeff = min_val  #
        loc = min_loc
    else:
        coeff = max_val
        loc = max_loc
    return coeff, loc


def get_temptable(centroids: np.ndarray):
    n_units = len(centroids)
    n_features = centroids.shape[1]
    collided_len = 2 * n_features - 1
    temptable = np.zeros([n_units, n_units, collided_len, n_features])
    for i in range(n_units):
        for j in range(n_units):
            for k in range(collided_len):
                # 两个centroids的平移组合
                if k <= n_features - 1:
                    temptable[i][j][k] = np.pad(centroids[i][-k - 1:], (0, n_features - k - 1)) + centroids[j]
                else:
                    temptable[i][j][k] = centroids[i] + np.pad(centroids[j][: collided_len - k],
                                                               (k + 1 - n_features, 0))
    return temptable


def template_matching_sqdiff(data, labels, centroids: np.ndarray):

    outlier_ids = np.copy(np.where(labels == -1)).flatten()
    outlier_snips = data[outlier_ids]
    n_features = data.shape[1]
    n_centroids = centroids.shape[0]
    renewed_laebls = np.copy(labels)

    for each in range(len(outlier_ids)):
        the_snip = outlier_snips[each]
        sqdiffs_centroid = np.array([])
        for each_c in range(len(centroids)):
            sqdiff = tm_sqdiff(the_snip, centroids[each_c])
            sqdiffs_centroid = np.append(sqdiffs_centroid, sqdiff)
        minsqdiff_id_centroid = np.argmin(sqdiffs_centroid)
        renewedlaebl = minsqdiff_id_centroid
        renewed_laebls[outlier_ids[each]] = renewedlaebl

    return renewed_laebls


def template_matching(data, labels, frames, centroids: np.ndarray, detect_threshold):
    '''
    match pursuit
    '''
    # 1. prepare
    outlier_ids = np.where(labels == -1)[0]
    outlier_snips = data[outlier_ids]
    n_features = data.shape[1]
    frames_before = frames_after = n_features // 2
    offset = 0  # ★
    renewed_laebls = np.copy(labels)
    renewed_frams = np.copy(frames)

    # 2. loop for each outlier
    for each in range(len(outlier_ids)):
        # 2.1 current outlier to process
        the_snip = outlier_snips[each]
        the_frame = frames[outlier_ids[each]]

        # 2.2 'sqdiff match' with centroids, get the best
        sqdiffs = np.array([])
        for i_c in range(len(centroids)):
            sqdiff = tm_sqdiff(the_snip, centroids[i_c])
            sqdiffs = np.append(sqdiffs, sqdiff)
        best_match = np.argmin(sqdiffs)

        # 2.3 substract the best_match centroid
        the_snip1 = the_snip - centroids[best_match]

        # 2.4 threshold detection for the residual
        best_match1 = None
        for i in range(len(the_snip1)):
            relative_frame = i
            current_val = the_snip1[i]
            abs_current_val = abs(current_val)
            n_before = i - frames_before
            if n_before < 0 :
                n_before = 0
            n_after = i + frames_after + 1
            residual_spike = the_snip1[n_before: n_after]
            if current_val < 0 and abs_current_val >= detect_threshold and abs_current_val == np.argmax(residual_spike):
                # 2.5  'sqdiff match' with centroids for the detected residual spike
                sqdiffs = np.array([])
                for i_c in range(len(centroids)):
                    sqdiff = tm_sqdiff(residual_spike, (centroids[i_c])[n_before: n_after])
                    sqdiffs = np.append(sqdiffs, sqdiff)
                best_match1 = np.argmin(sqdiffs)
                the_frame1 = the_frame - (n_features // 2 - relative_frame)
                break
        # 2.6 recover label and frame
        if best_match1 is not None:
            if the_frame1 <= the_frame:
                frame_pair = [the_frame1, the_frame]
                label_pair = [best_match1, best_match]
            else:
                frame_pair = [the_frame, the_frame1]
                label_pair = [best_match, best_match1]
            renewed_laebls = np.delete(renewed_laebls, outlier_ids[each] + offset)
            renewed_laebls = np.insert(renewed_laebls, outlier_ids[each] + offset, label_pair)
            # 更新 frames
            renewed_frams = np.delete(renewed_frams, outlier_ids[each] + offset)
            renewed_frams = np.insert(renewed_frams, outlier_ids[each] + offset, frame_pair)
            offset += 1
        else:
            renewed_laebls[outlier_ids[each] + offset] = best_match
    return [renewed_laebls, renewed_frams]


def main():
    np.set_printoptions(threshold=np.infty)

    ######################################################
    # data extract & preprocess
    ######################################################
    recording, sorting_true = load_data(globvar.h5data_path)
    extrt_dict, recording_bp = preprocessing(recording, sorting_true)
    channel_location: list = extrt_dict['channel_loc']

    ######################################################
    # threshold detection
    ######################################################
    detec = Detection(extrt_dict)
    split_list = globvar.h5filename.split('_')
    time_buff = []

    start = time.time()
    detct_rst = detec.mc_run()  # detection
    # divide-and-conquer for multi-electorde
    merged_frames, grouped_frames, grouped_frameids = detec.get_frame_groups(detct_rst)
    time_buff.append(time.time() - start)

    # evaluate detection results
    fn_frames, fp_frames, evalinfo_dict, according_labels = detec.evaluate(merged_frames)
    print('----------------detection fnished----------------')
    print(
        f"precision: {evalinfo_dict['precision']}, recall: {evalinfo_dict['recall']}, F1: {evalinfo_dict['F1_score']}")

    start = time.time()
    snipgroups = detec.get_grouped_snippets(grouped_frames)
    # -----remove empty array-----
    for i in np.arange(len(grouped_frames))[::-1]:
        if len(grouped_frames[i]) == 0:
            del grouped_frames[i]
            del grouped_frameids[i]
            del snipgroups[i]
            del channel_location[i]
    time_buff.append(time.time() - start)
    for i in range(len(grouped_frames)):
        print(f"group{i}, frames{grouped_frames[i].shape}")

    ######################################################
    # KNN outlier detection (NOT USED)
    ######################################################
    all_snippets = np.zeros([0, extrt_dict['snip_frame_before'] + extrt_dict['snip_frame_after'] + 1])
    for i in range(len(snipgroups)):
        all_snippets = np.vstack((all_snippets, snipgroups[i]))
    outlier_marks = np.zeros(len(all_snippets), dtype=np.int)
    grouped_outlier_marks = []
    cnt_sample = 0
    for i in range(len(snipgroups)):
        n_samples = len(snipgroups[i])
        grouped_outlier_marks.append(outlier_marks[cnt_sample: cnt_sample + n_samples])
        cnt_sample += n_samples

    ######################################################
    # PCA for each group
    ######################################################
    start = time.time()
    pca_snipgroups_rst = []
    p = Pool(min(globvar.n_multiprocess, len(snipgroups)))
    for i in range(len(snipgroups)):
        pca_snipgroups_rst.append(p.apply_async(decomposition, args=(snipgroups[i], globvar.group_min_samples, )))
    p.close()
    p.join()
    pca_snipgroups = []
    for each in pca_snipgroups_rst:
        pca_snipgroups.append(each.get())
    time_buff.append(time.time() - start)

    # get benign data
    pca_benign_snipgroups = []
    benign_snipgroups = []
    benign_framegroups = []
    for i in range(len(grouped_outlier_marks)):
        ids = np.where(grouped_outlier_marks[i] != -1)[0]
        benign_snipgroups.append((snipgroups[i])[ids])
        benign_framegroups.append((grouped_frames[i])[ids])
        pca_benign_snipgroups.append((pca_snipgroups[i])[ids])

    ######################################################
    # cluster
    ######################################################
    start = time.time()
    # multiprocess pool
    p = Pool(min(globvar.n_multiprocess, len(benign_snipgroups)))
    mp_cluster_rst = []
    for i_group in range(len(benign_snipgroups)):
        mp_cluster_rst.append(p.apply_async(cluster, args=(benign_snipgroups[i_group], pca_benign_snipgroups[i_group],)))
    p.close()
    p.join()
    benign_labelgroup_list = []  # labels of benign samples
    tempgroups_list = []  # cluster centroids
    for i_group in range(len(benign_snipgroups)):
        i_cluster_rtn = (mp_cluster_rst[i_group]).get()
        benign_labelgroup_list.append(i_cluster_rtn[0])  # get labels
        tempgroups_list.append(i_cluster_rtn[1])  # get centroids

    labelgroup_list = copy.deepcopy(grouped_outlier_marks)  # ★★★labels
    for i in range(len(labelgroup_list)):
        benign_ids = np.where(grouped_outlier_marks[i] != -1)[0]
        labelgroup_list[i][benign_ids] = benign_labelgroup_list[i]  # fill labels
    time_buff.append(time.time() - start)
    #
    for i in range(len(labelgroup_list)):
        statistical = Counter(labelgroup_list[i])
        print(statistical)

    ######################################################
    # postprocessing
    ######################################################
    # ----------------templates merging------------------
    start = time.time()
    # merging
    merged_centroids, merged_labelgroup_list = templates_merging(tempgroups_list,
                                                                 labelgroup_list, channel_location=channel_location,
                                                                 channel_type=(split_list[2].split("-"))[0],
                                                                 sigma=globvar.sigma, peek_diff=globvar.peek_diff)
    time_buff.append(time.time() - start)

    ######################################################
    # tempalte matching
    ######################################################
    start = time.time()
    p = Pool(min(globvar.n_multiprocess, len(merged_labelgroup_list)))
    aftertm_list = []
    threshold = np.mean(extrt_dict['stdvir_channels']) * globvar.detect_th
    for i_group in range(len(merged_labelgroup_list)):
        aftertm_list.append(p.apply_async(
            template_matching, args=(snipgroups[i_group], merged_labelgroup_list[i_group], grouped_frames[i_group], merged_centroids, threshold,)))
    p.close()
    p.join()
    recovered_label_list = []
    recovered_frame_list = []
    # recovered_frame_list = grouped_frames
    for i_group in range(len(aftertm_list)):
        # recovered_label_list.append(aftertm_list[i_group])
        tm_rtn = (aftertm_list[i_group]).get()
        recovered_label_list.append(tm_rtn[0])
        recovered_frame_list.append(tm_rtn[1])
    time_buff.append(time.time() - start)
    ######################################################
    # get performance
    ######################################################
    recovered_labels = np.array([])
    recovered_frames = np.array([])
    for i in range(len(recovered_label_list)):
        recovered_labels = np.append(recovered_labels, recovered_label_list[i])
        recovered_frames = np.append(recovered_frames, recovered_frame_list[i])

    sorting = se.NumpySortingExtractor()
    sorting.set_sampling_frequency(extrt_dict['sample_freq'])
    sorting.set_times_labels(recovered_frames, recovered_labels)
    n_units = len(sorting.get_unit_ids())

    # compare to ground-truth
    comp: sc.GroundTruthComparison = sc.compare_sorter_to_ground_truth(sorting_true, sorting, delta_time=0.4, match_mode='best')
    # get_performance
    comp.print_performance(method='by_unit')
    comp.print_performance()
    gt_num_units = len(sorting_true.get_unit_ids())
    tested_num_units = len(sorting.get_unit_ids())
    print(f"GT num_units: {gt_num_units}")
    print(f"tested num_units: {tested_num_units}")
    perf = comp.get_performance(method='pooled_with_average', output='pandas')
    perf['gt_num_units'] = gt_num_units
    perf['tested_num_units'] = tested_num_units
    perf['time'] = sum(time_buff)
    perf['dataset'] = globvar.h5filename
    # output to csv
    perf.to_csv("perf_logger.csv", mode='a', header=False)

    assert True  # debug


if __name__ == '__main__':
    main()
