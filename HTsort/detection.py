import numpy as np
from scipy.io import loadmat, savemat
import globvar
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter
import copy
import csv
from sklearn.decomposition import PCA, KernelPCA
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process, Pool
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw


class Detection(object):
    def __init__(self, extractor_dict: dict, threshold_coeff=globvar.threshold_coeff):
        self._traces = extractor_dict['traces']
        self._spike_labels = extractor_dict['spike_labels_GT'].astype(np.int)
        self._spike_times = extractor_dict['spike_times_GT'].astype(np.int)  # i.e. spike_frames_GT
        # parameters
        self.n_channels = extractor_dict['n_channels']
        self.n_units = extractor_dict['n_units']
        self.frame_before = extractor_dict['detect_frame_before']
        self.frame_after = extractor_dict['detect_frame_after']
        self.snipframe_before = extractor_dict['snip_frame_before']
        self.snipframe_after = extractor_dict['snip_frame_after']
        self.snippets_len = self.frame_before + self.frame_after + 1
        self.threshold_voltagevalue = threshold_coeff * extractor_dict['stdvir_channels']

    # detection for multi-channel
    def mc_run(self, n_mp=globvar.n_multiprocess):
        '''
        Parameters
        -------------
        n_mp: int
            Number of parallel processes allowed

        Returns
        --------------
        mpdect_rst: list
            Contains n_channels sublists, each containing two np 1D array elements,
             respectively the frames detected by the channel and the amplitude of
              the corresponding point
        '''
        p = Pool(min(n_mp, self.n_channels))
        rst = []
        for ch_id in range(self.n_channels):
            rst.append(p.apply_async(self.run, args=(self._traces[ch_id], ch_id)))
        p.close()
        p.join()
        mpdect_rst = []
        for each in rst:
            mpdect_rst.append(each.get())
        return mpdect_rst

    # for each channel
    def run(self, trace: np.ndarray, ch_id: int):
        '''
        detection for one channel

        Parameters
        ----------------------
        trace: np.ndarray
            1 D array of trace
        ch_id: int
        `   channel id

        Returns
        -----------------------
        dect_rest: list
            list contains detected_frames, detected_frame_ampls
        '''
        # data_len = len(trace)
        # output
        detected_frames = np.zeros(0, dtype=np.int)  # spike times array
        detected_frame_ampls = np.zeros(0)  # amplitude

        frames_1 = np.where(trace <= -self.threshold_voltagevalue[ch_id])[0]
        ampl_frames_1 = trace[frames_1]
        for i in range(len(frames_1)):
            frame = frames_1[i]
            snippet = trace[frame - self.frame_before: frame + self.frame_after + 1]
            if np.abs(ampl_frames_1[i]) == np.max(np.abs(snippet)):
                detected_frames = np.append(detected_frames, frame)
                detected_frame_ampls = np.append(detected_frame_ampls, np.abs(ampl_frames_1[i]))

        print(f'channel{ch_id}, frames{detected_frames.shape}')
        dect_rst = [detected_frames, detected_frame_ampls]
        return dect_rst

    # not used
    def run_alpha(self, trace: np.ndarray, ch_id: int):
        data_len = len(trace)
        # output
        detected_frames = np.zeros(0, dtype=np.int)
        detected_frame_ampls = np.zeros(0)

        for frame in range(data_len):
            current_value = trace[frame]
            abs_current_value = abs(current_value)
            snippet = trace[frame - self.frame_before: frame + self.frame_after + 1]
            if current_value < 0 and abs_current_value >= self.threshold_voltagevalue[ch_id] and abs_current_value == np.max(np.abs(snippet)):  # 是过阈值的局部极值点
                detected_frames = np.append(detected_frames, frame)
                detected_frame_ampls = np.append(detected_frame_ampls, abs_current_value)

        print(f'channel{ch_id}, frames{detected_frames.shape}')
        dect_rst = [detected_frames, detected_frame_ampls]
        return dect_rst

    # divide-and-conquer for multi-electorde
    def get_frame_groups(self, mpdect_rst: list):
        '''
        Parameters
        -----------------
        mpdect_rst: list
            Contains n_channels sublists, each containing two np 1D array elements,
             respectively the frames detected by the channel and the amplitude of
              the corresponding point

        Returns
        ---------------
        merged_frames： 1D array
        grouped_frames： list
        grouped_frameids:  list
        '''
        merged_frames = np.array([], dtype=np.int)
        grouped_frames = []
        grouped_frameids = []
        frame_len_eachchan = np.array([])
        # second_maxampl = []
        for i in range(self.n_channels):
            # grouped_frames = np.append(grouped_frames, np.array([0]))
            grouped_frames.append(np.array([], dtype=np.int))
            grouped_frameids.append(np.array([], dtype=np.int))
            frame_len_eachchan = np.append(frame_len_eachchan, len(mpdect_rst[i][0]))
            # second_maxampl.append(np.array([], dtype=dict))  # dict keys: 'chid', 'frame'
        pointers = np.zeros(self.n_channels, dtype=np.int)
        cnt = 0
        while (pointers != frame_len_eachchan).any():
            frames = np.array([])
            amplitudes = np.array([])
            for i in range(self.n_channels):
                if pointers[i] >= frame_len_eachchan[i]:
                    frames = np.append(frames, np.inf)
                    amplitudes = np.append(amplitudes, 0.)
                else:
                    frames = np.append(frames, (mpdect_rst[i][0])[pointers[i]])
                    amplitudes = np.append(amplitudes, (mpdect_rst[i][1])[pointers[i]])
            sorted_frames_id = (np.argsort(frames))[::-1]
            min_frame = frames[sorted_frames_id[-1]]
            for j in range(self.n_channels):
                if frames[sorted_frames_id[j]] - min_frame <= globvar.dect_frame_bias:
                    near_chan_ids = sorted_frames_id[j:]
                    pointers[near_chan_ids] += 1
                    maxampl_chanid = near_chan_ids[np.argmax(amplitudes[near_chan_ids])]
                    the_frame = int(frames[maxampl_chanid])
                    # if len(near_chan_ids) >= 2:
                    #     second_maxampl_chanid = near_chan_ids[np.argsort(amplitudes[near_chan_ids])[-2]]
                    #     second__the_frame = int(frames[second_maxampl_chanid])
                    #     dic: dict = {'ch_id': second_maxampl_chanid, 'frame': second__the_frame}
                    # else:
                    #     dic = {'ch_id': maxampl_chanid, 'frame': the_frame}
                    merged_frames = np.append(merged_frames, the_frame)
                    grouped_frameids[maxampl_chanid] = np.append(grouped_frameids[maxampl_chanid], cnt)
                    cnt += 1
                    grouped_frames[maxampl_chanid] = np.append(grouped_frames[maxampl_chanid], the_frame)

                    # second_maxampl[maxampl_chanid] = np.append(second_maxampl[maxampl_chanid], dic)
                    break

        return merged_frames, grouped_frames, grouped_frameids

    # evaluate detection results
    def evaluate(self, detection_frames: np.ndarray):
        fn_frames = np.array([], dtype=np.int)
        fp_frames = np.array([], dtype=np.int)
        according_labels = np.array([], dtype=np.int)
        ptr_GT = ptr_dect = 0
        tp = 0  # true pos
        dect_len = len(detection_frames)
        GT_len = len(self._spike_times)
        while True:
            if ptr_dect == dect_len and ptr_GT != GT_len:
                fn_frames = np.append(fn_frames, self._spike_times[ptr_GT:])
                break
            elif ptr_dect != dect_len and ptr_GT == GT_len:
                fp_frames = np.append(fp_frames, detection_frames[ptr_dect:])
                according_labels = np.append(according_labels, np.ones(dect_len - ptr_dect) .astype(np.int) * -1)
                break
            elif ptr_dect == dect_len and ptr_GT == GT_len:
                break
            else:
                frame_dect = detection_frames[ptr_dect]
                frame_GT = self._spike_times[ptr_GT]
                if frame_dect - frame_GT > globvar.dect_frame_bias:
                    fn_frames = np.append(fn_frames, frame_GT)
                    ptr_GT += 1
                elif frame_GT - frame_dect > globvar.dect_frame_bias:
                    fp_frames = np.append(fp_frames, frame_dect)
                    according_labels = np.append(according_labels, -1)
                    ptr_dect += 1
                elif abs(frame_GT - frame_dect) <= globvar.dect_frame_bias:
                    tp += 1
                    according_labels = np.append(according_labels, self._spike_labels[ptr_GT])
                    ptr_dect += 1
                    ptr_GT += 1
        fn = len(fn_frames)
        fp = len(fp_frames)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = 2 * recall * precision / (recall + precision)
        eval_dict = {'recall': recall, 'precision': precision, 'tp': tp, 'fn': fn, 'fp': fp, 'F1_score': f1_score}
        return fn_frames, fp_frames, eval_dict, according_labels

    def get_grouped_snippets(self, groupedframes: list):
        n_groups = len(groupedframes)
        snipframe_range = self.snipframe_before + self.snipframe_after + 1
        grouped_snippets = []
        len_log = np.array([])
        for n_group in range(n_groups):
            length = len(groupedframes[n_group])
            len_log = np.append(len_log, length)
            if length == 0:
                grouped_snippets.append(np.array([]))
                continue
            snipbuff = np.zeros([0, snipframe_range])
            for each_frame in groupedframes[n_group]:
                # print(f"{n_group}group,{each_frame}frame")
                snippet = (self._traces[n_group])[each_frame - self.snipframe_before: each_frame + self.snipframe_after + 1]
                # print(f"snippet shape {snippet.shape}")
                if len(snippet) == 0:
                    id_frame = np.where(groupedframes[n_group] == each_frame)[0]
                    groupedframes[n_group] = np.delete(groupedframes[n_group], np.arange(id_frame, len(groupedframes[n_group])))
                    break
                snipbuff = np.vstack((snipbuff, snippet.reshape([1, -1])))
            grouped_snippets.append(snipbuff)

        return grouped_snippets

    def get_snippets(self, groupedframes: list, channel_loccation):
        n_groups = len(groupedframes)
        snip_range = self.snipframe_before + self.snipframe_after + 1
        snippets = np.zeros([0, snip_range + 2])
        # snippets = np.zeros([0, snip_range])
        frames = np.array([])
        for i_group in range(n_groups):
            frames_array: np.ndarray = groupedframes[i_group]
            loc = channel_loccation[i_group]
            n_frames = len(frames_array)
            if n_frames != 0:
                for each_frame in frames_array:
                    snip = np.append((self._traces[i_group])[each_frame - self.snipframe_before: each_frame + self.snipframe_after + 1], loc[1])
                    snip = np.insert(snip, 0, loc[0])
                    # snip = (self._traces[i_group])[each_frame - self.snipframe_before: each_frame + self.snipframe_after + 1]
                    snippets = np.vstack((snippets, snip))
                    frames = np.append(frames, each_frame)
        sort_ids = np.argsort(frames)
        sort_snippets = snippets[sort_ids]
        return sort_snippets

    def __search(self, value, array):
        length = len(array)
        lo = 0
        hi = length - 1
        idx = self.__binary_search(lo, hi, value, array)
        return idx

    def __binary_search(self, lo, hi, value, array):
        mid = (lo + hi) // 2
        if lo == hi:
            if array[mid] == value:
                return mid
            else:
                if mid == 0:
                    return mid if abs(array[mid] - value) < abs(array[mid + 1] - value) else (mid + 1)
                elif mid == len(array) - 1:
                    return mid if abs(array[mid] - value) < abs(array[mid - 1] - value) else (mid - 1)
                else:
                    a1 = array[mid - 1: mid + 2]
                    a2 = np.ones(3) * value
                    diff = np.abs(a2 - a1)
                    nearest_id = np.argmin(diff) + mid - 1
                    return nearest_id
        if array[mid] == value:
            return mid
        elif array[mid] < value:
            return self.__binary_search(mid + 1, hi, value, array)
        else:
            return self.__binary_search(lo, mid, value, array)


def save_something_as_txt(content, filename="new_file"):
    full_filename = filename + ".txt"
    with open(full_filename, "a+") as fob:
        fob.write(content)


# PCA
def decomposition(data, min_samples=0):
    if len(data) <= min_samples:
        return np.zeros([len(data), globvar.n_components])
    reductor = PCA(n_components=globvar.n_components)
    decomp_data = reductor.fit_transform(data)
    # print("pca.n_components_: {}".format(pca.n_components_))
    # print("decomp_data.shape: {}".format(decomp_data.shape))
    return decomp_data















