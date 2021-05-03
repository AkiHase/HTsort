import numpy as np
import matplotlib.pyplot as plt
import os
import time
import globvar
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import MEArec as mr
from spikeforest2_utils import AutoRecordingExtractor, AutoSortingExtractor


def load_data(filepath):
    '''
    Parameters
    -----------
    filepath: str

    Returns
    -----------
    prep_recording: RecordingExtractor
        preprocessed RecordingExtractor
    sorting_GT: SortingExtractor
    '''
    recording = se.MEArecRecordingExtractor(filepath)
    sorting_GT = se.MEArecSortingExtractor(filepath)

    # recording info
    fs = recording.get_sampling_frequency()
    channel_ids = recording.get_channel_ids()
    channel_loc = recording.get_channel_locations()
    num_frames = recording.get_num_frames()
    duration = recording.frame_to_time(num_frames)
    print(f'Sampling frequency:{fs}')
    print(f'Channel ids:{channel_ids}')
    print(f'channel location:{channel_loc}')
    print(f'frame num:{num_frames}')
    print(f'recording duration:{duration}')
    # sorting_GT info
    unit_ids = sorting_GT.get_unit_ids()
    print(f'unit ids:{unit_ids}')
    return recording, sorting_GT


def load_spikeforest_data(recording_path: str, sorting_true_path: str, download=True):
    recording = AutoRecordingExtractor(recording_path, download=download)
    sorting_GT = AutoSortingExtractor(sorting_true_path)
    # recording info
    fs = recording.get_sampling_frequency()
    channel_ids = recording.get_channel_ids()
    channel_loc = recording.get_channel_locations()
    num_frames = recording.get_num_frames()
    duration = recording.frame_to_time(num_frames)
    print(f'Sampling frequency:{fs}')
    print(f'Channel ids:{channel_ids}')
    print(f'channel location:{channel_loc}')
    print(f'frame num:{num_frames}')
    print(f'recording duration:{duration}')
    # sorting_GT info
    unit_ids = sorting_GT.get_unit_ids()
    print(f'unit ids:{unit_ids}')
    return recording, sorting_GT


def preprocessing(recording: se.MEArecRecordingExtractor, sorting: se.MEArecSortingExtractor):
    info_dict = {}
    # bandpass filter
    recording_f = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=3000)
    # recording_f = st.preprocessing.resample(recording_f1, recording_f1.get_sampling_frequency() // 2)  # down sampling
    # recording_f = st.preprocessing.common_reference(recording_f, reference='median')

    # saveinfo
    # info_dict['sorting'] = sorting
    info_dict['n_channels'] = n_channels = recording_f.get_num_channels()
    info_dict['channel_loc'] = channel_location = recording_f.get_channel_locations()
    info_dict['n_frames'] = n_frames = recording_f.get_num_frames()
    info_dict['n_units'] = n_units = len(sorting.get_unit_ids())
    info_dict['sample_freq'] = recording_f.get_sampling_frequency()
    info_dict['traces'] = traces = recording_f.get_traces()
    info_dict['detect_frame_before'] = int(recording_f.time_to_frame(globvar.detect_ms_before * 0.001))
    info_dict['detect_frame_after'] = int(recording_f.time_to_frame(globvar.detect_ms_after * 0.001))
    info_dict['snip_frame_before'] = int(recording_f.time_to_frame(globvar.snippet_ms_before * 0.001))
    info_dict['snip_frame_after'] = int(recording_f.time_to_frame(globvar.snippet_ms_after * 0.001))

    # get (estimate of stdvir) = (median(abs(S))//0.6745) for each channel
    stdvirs = np.array([])
    for i in range(n_channels):
        stdvirs = np.append(stdvirs, np.median(np.abs(traces[i])) / 0.6745)
    info_dict['stdvir_channels'] = stdvirs
    # get spike_labels_GT and spike_times_GT
    pointers = np.zeros(n_units).astype(np.int16)
    trains_buff = []
    n_spike_frames = 0
    for i in range(n_units):
        spike_train = sorting.get_unit_spike_train(unit_id=i)
        spike_train = np.append(spike_train, np.inf)
        trains_buff.append(spike_train)
        n_spike_frames += len(spike_train) - 1
    spike_labels_GT = np.array([])
    spike_times_GT = np.array([])
    for i in range(n_spike_frames):
        frame_buff = np.array([])
        #
        for j in range(n_units):
            frame_buff = np.append(frame_buff, (trains_buff[j])[pointers[j]])
        min_id = np.argmin(frame_buff)
        # print(min_id)
        spike_labels_GT = np.append(spike_labels_GT, min_id)
        spike_times_GT = np.append(spike_times_GT, frame_buff[min_id])
        pointers[min_id] += 1
    info_dict['spike_labels_GT'] = spike_labels_GT.astype(np.int)
    info_dict['spike_times_GT'] = spike_times_GT.astype(np.int)
    return info_dict, recording_f


if __name__ == "__main__":
    # recgen = mr.load_recordings(globvar.h5data_path)
    # mr.plot_recordings(recgen)
    # mr.plot_templates(recgen)
    # mr.plot_waveforms(recgen)
    # mr.plot_amplitudes(recgen)

    recording, sorting = load_data(globvar.h5data_path)
    info_dict, recording_f = preprocessing(recording, sorting)
    # sw.plot_unit_waveforms(recording_f, sorting, max_spikes_per_unit=10000000,ms_before=globvar.snippet_ms_before, ms_after=globvar.snippet_ms_after)
    channel_loc = recording_f.get_channel_locations()
    print(f'channel location:{channel_loc}')
    sw.plot_electrode_geometry(recording_f)
    plt.show()
    print(recording)
