import time
import os
import torch
from tqdm import tqdm
import numpy as np
import scipy.io as scp
from bisect import bisect_left
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset

from layers.graph import Graph, HeteroGraph

def get_frame_instance_dict_new(tracks):
    """
    Read data from tracks and return a dataset list and a list of sorted frame id:
    [   -> this length equals the number of datasets
        {frame_id:
            {vehicle_id:
                # 6 features
                [dataset_id, vehicle_id, frame_id, local_x, local_y, lane_id]
            }
        }
    ]
    the form of the input tracks like:
    [   -> equals the number of datasets
        [ -> equals the number of unique vehicle ids
            [
                [frame_id, ...],
                [local_x, ...],
                [local_y, ...],
                [lane_id, ...]
            ]

        ]

    ]
    
    """
    # for unique frame id among multiple datasets.
    # Note: dataset_id starts from 1, ends at 6

    dataset_list = []
    sorted_frame_id_list = []
    for dataset_idx, dataset in enumerate(tracks):
        dataset_id = dataset_idx + 1
        frame_dict = dict()
        for vehicle_idx, vehicle_data in enumerate(dataset):
            if len(vehicle_data) <= 1:
                continue
            # assert vehicle_data.shape[0] == 4
            vehicle_id = vehicle_idx + 1
            
            for frame_id, local_x, local_y, lane_id in vehicle_data.transpose():
                vehicle_dict = frame_dict.get(frame_id, {})
                vehicle_dict[int(vehicle_id)] = np.array([dataset_id, vehicle_id, frame_id, local_x, local_y, lane_id], dtype=np.float32)
                frame_dict[int(frame_id)] = vehicle_dict
        dataset_list.append(frame_dict)
        sorted_frame_id_list.append(sorted(frame_dict.keys()))
    return dataset_list, sorted_frame_id_list

class NgsimFeeder(Dataset):
    def __init__(self, mat_fpath, graph_args={}, t_h=30, t_f=50, down_sampling=2, neighbor_distance=90, max_num_object=120, train_val_test='train', use_hetero_graph=False, **kwargs) -> None:
        self.data_path = mat_fpath
        self.history_frames = kwargs.get('t_h', t_h) # length of track history, 3 seconds with a sampling rate of 10 Hz
        self.future_frames = kwargs.get('t_f', t_f) # length of predicted trajectory, 5 seconds with a sampling rate of 10 Hz
        self.total_frames = t_h + t_f
        self.down_sampling_steps = kwargs.get('down_sampling_steps', down_sampling)
        self.neighbor_distance = kwargs.get('neighbor_distance', neighbor_distance) # the unit of it is the feet.
        self.max_num_object = kwargs.get('max_num_object', max_num_object)
        self.data_aug_ratio = kwargs.get('data_aug_ratio', 0.0)
        self.train_val_test = train_val_test
        self.num_object_record = self.max_num_object
        self.use_hetero_graph = use_hetero_graph
        self.max_len_history = self.history_frames // self.down_sampling_steps + 1
        self.max_len_future = self.future_frames // self.down_sampling_steps
        if self.use_hetero_graph:
            graph_args = kwargs.get('graph_args', {})
            self.num_hetero_types = graph_args.get('num_hetero_types', 3)
            
        
        self._load_and_process_data()

    def _load_and_process_data(self):
        
        # load data from .mat format file.
        mat_data = scp.loadmat(self.data_path)
        # self.traj: shape of (*, 47), 1: Dataset Id 2: Vehicle Id 3: Frame Id 4: Local X 5: Local Y 6: Lane Id 7: Lateral maneuver 8: Longitudinal maneuver 9-47: Neighbor Car Ids at grid location
        # self.tracks: shape of (num_dataset, *), where num_dataset = 6
        self.traj, self.tracks = mat_data['traj'], mat_data['tracks']
        self.length = len(self.traj) 
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):

        dset_id, veh_id, cur_frame_id = self.traj[index, :3].astype(int).tolist()
        neighbor_grid = self.traj[index, 8:]
        object_frame_feature = np.zeros((self.max_num_object, self.total_frames // 2 + 1, 7)) # (V, T, C), T = 41 = 16 (history) + 25 (future)
        

        object_frame_feature, valid_vehicle_num, mean_xy = self._process_data(
            cur_frame_id, veh_id, dset_id, neighbor_grid, object_frame_feature
        ) # object_frame_feature: (V, T, C)
    
        neighbor_matrix = np.zeros((self.max_len_history, self.max_num_object, self.max_num_object), dtype=np.int8)
        for cur_step in range(self.max_len_history):
            visible_object_value = object_frame_feature[:, cur_step, :]
            for i in range(valid_vehicle_num):
                for j in range(i+1):
                    longitu_i, lane_i = visible_object_value[i][4:6]
                    longitu_j, lane_j = visible_object_value[j][4:6]
                    if abs(longitu_i - longitu_j) < self.neighbor_distance and abs(lane_i - lane_j) < 2.0:
                        neighbor_matrix[cur_step, i, j] = neighbor_matrix[cur_step, j, i] = 1

        if self.use_hetero_graph:
            if self.num_hetero_types == 3:
                neighbor_matrix = self._fill_hetero_type(neighbor_matrix, visible_object_value)
            elif self.num_hetero_types == 6:
                neighbor_matrix = self._fill_hetero_type_more(neighbor_matrix, visible_object_value)
            else:
                raise ValueError('Valid value of num_hetero_types, which should be 3 or 6.')
        
        object_frame_feature = object_frame_feature.transpose(2, 1, 0) # (C, T, V)
        # data aumentation
        if self.train_val_test == 'train' and np.random.random() < self.data_aug_ratio:
            angle = 2 * np.pi * np.random.random()
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)

			# randomly rotate x, y and now_mean_xy
            angle_mat = np.array(
				[[cos_angle, -sin_angle],
				[sin_angle, cos_angle]])

            xy = object_frame_feature[3:5, :, :]
            num_xy = np.sum(xy.sum(axis=0).sum(axis=0) != 0) # get the number of valid data

            # angle_mat: (2, 2), xy: (2, 7, max_num_object)
            out_xy = np.einsum('ab,btv->atv', angle_mat, xy)
            mean_xy = np.matmul(angle_mat, mean_xy)
            xy[:, :, :num_xy] = out_xy[:, :, :num_xy]
            object_frame_feature[3:5, :, :] = xy

        return object_frame_feature, neighbor_matrix, mean_xy, valid_vehicle_num
        

    def _process_data(self, cur_frame_id, veh_id, dset_id, neighbor_grid, object_frame_feature):
        # object_frame_feature: shape of (V, T, C)
        observed_veh_ids = [veh_id] + neighbor_grid.astype(int).tolist()
        valid_num = 0
        for i, v_id in enumerate(observed_veh_ids):
            track = self._get_tracks(v_id, cur_frame_id, dset_id)
            if track is None:
                assert i != 0  # the target vehicle's history track can not be None
                continue
            history, future = track
            assert i > 0 or (i == 0 and len(future) > 0 and len(history) > 0)
            history_start = self.max_len_history - len(history)
            object_frame_feature[valid_num, history_start: self.max_len_history, 2: 6] = history # [[frame_id, local_x, local_y, lane_id], ...]
            # we add a "mark" feature to the object_frame_feature to indicate whether this frame will not be included into loss calculation.
            # object_frame_feature[valid_num, :history_start, -1] = 0
            object_frame_feature[valid_num, history_start: self.max_len_history, -1] = 1

            future_end = self.max_len_history + len(future)
            object_frame_feature[valid_num, self.max_len_history: future_end, 2: 6] = future
            #TODO as above, but if the length of track history is less than self.t_h // 2, we will ignore this future track in the loss calculation but it will be considered in the interaction with other vehicles.
            # object_frame_feature[valid_num, future_end:, -1] = 0
            object_frame_feature[valid_num, self.max_len_history: future_end, -1] = 1 if len(history) > 0 or valid_num == 0 else 0 # (self.history_frames // self.down_sampling_steps // 2)
            # in the end, we specify dataset_id and vehicle_id in object_frame_feature.
            object_frame_feature[valid_num, :, 0: 2] = [dset_id, v_id]
            valid_num += 1

        # compute the mean values of x and y for zero-centralization
        mean_xy = object_frame_feature[:valid_num, self.max_len_history-1, 3: 5].mean(axis=0) # shape of (2, )
        object_frame_feature[object_frame_feature[:, :, 3: 5].sum(axis=-1) > 0, 3: 5] -= mean_xy
        
        return object_frame_feature, valid_num, mean_xy
    
    def _get_tracks(self, veh_id, observed_frame_id, dset_id):

        if veh_id == 0 or self.tracks.shape[1] <= veh_id - 1:
            return None

        track = self.tracks[dset_id-1][veh_id-1].transpose() # (*, 4), the elements in second dimension are: frame_id, local_x, local_y, lane_id
        # get the row position of observed_frame_id in track.
        frame_index = np.argwhere(track[:, 0] == observed_frame_id)
        if track.size == 0 or frame_index.size == 0:
            return None

        frame_index = frame_index.item() 
        history_start = max(0, frame_index - self.history_frames)
        history = track[history_start: frame_index+1: self.down_sampling_steps] # (<= 16, 4)
        
        future_end = min(len(track), frame_index + self.future_frames + 1)
        future_start = frame_index + self.down_sampling_steps
        future = track[future_start: future_end: self.down_sampling_steps] # (<= 25, 4)
        # his_and_fut_feature = track[history_start: future_end: self.down_sampling_steps] # (40, 4)
        
        return history, future

    def _fill_hetero_type_more(self, neighbor_matrix, visible_object_value):
        neighbor_type_matrix = neighbor_matrix
        for i in range(visible_object_value.shape[0]):
            for j in range(visible_object_value.shape[0]):
                if neighbor_matrix[i][j] == 1:
                    local_y_i, lane_id_i = visible_object_value[i][4: 6] # local_y and lane_id are the fifth and six-th feature in visible_object_value respectively.
                    local_y_j, lane_id_j = visible_object_value[j][4: 6]
                    if local_y_i > local_y_j:
                        if lane_id_i < lane_id_j:
                            neighbor_type_matrix[i][j], neighbor_type_matrix[j][i] = 1, 4
                        elif lane_id_i > lane_id_j:
                            neighbor_type_matrix[i][j], neighbor_type_matrix[j][i] = 3, 6
                        else:
                            neighbor_type_matrix[i][j], neighbor_type_matrix[j][i] = 2, 5

        return neighbor_type_matrix

    def _fill_hetero_type(self, neighbor_matrix, visible_object_value):
        neighbor_type_matrix = neighbor_matrix
        for i in range(visible_object_value.shape[0]):
            for j in range(0, i+1):
                if neighbor_matrix[i][j] == 1:
                    lane_id_i = visible_object_value[i][5] # lane_id is the six-th feature in visible_object_value
                    lane_id_j = visible_object_value[j][5]
                    if lane_id_i > lane_id_j:
                        neighbor_type_matrix[i][j], neighbor_type_matrix[j][i] = self.num_hetero_types-1, self.num_hetero_types
                    elif lane_id_i < lane_id_j:
                        neighbor_type_matrix[i][j], neighbor_type_matrix[j][i] = self.num_hetero_types, self.num_hetero_types-1

                    else:
                        neighbor_type_matrix[i][j] = neighbor_type_matrix[j][i] = 1

        return neighbor_type_matrix
        

def get_frame_instance_dictII(traj):
    """
    Read data from traj and return a dictionary:
    {dataset_id
        {frame_id:
            {object_id:
                # 47 features
                [dataset_id, vehicle_id, frame_id, local_x, local_y, lane_id, ...]
            }
        }
    }
    
    """
    # for unique frame id among multiple datasets.
    # Note: dataset_id starts from 1, ends at 6

    now_dict = {}

    for row in traj:
        # row[0]: dataset_id, row[1]: vehicle_id, row[2]: frame_id
        # row[2] =+ frame_counts[int(row[0])-1]
        no_dict = now_dict.get(int(row[0]), {})
        n_dict = no_dict.get(int(row[2]), {})
        n_dict[int(row[1])] = row
        no_dict[int(row[2])] = n_dict
        now_dict[int(row[0])] = no_dict

    return now_dict

def get_frame_instance_dict(traj):
    """
    Read data from traj and return a dictionary:
    {dataset_id
        {frame_id:
            {vehicle_id:
                # 6 features
                [dataset_id, vehicle_id, frame_id, local_x, local_y, lane_id]
            }
        }
    }
    
    """
    # for unique frame id among multiple datasets.
    # Note: dataset_id starts from 1, ends at 6
    
    '''
    dataset_dict = defaultdict(lambda: 0)
    for row in traj:
        dataset_dict[int(row[0])-1] += 1 
    frame_counts = [0] 
    for i in range(len(dataset_dict)-1):
        frame_counts.append(dataset_dict[i] + frame_counts[-1])
    '''

    now_dict = {}

    for row in traj:
        # row[0]: dataset_id, row[1]: vehicle_id, row[2]: frame_id
        # row[2] =+ frame_counts[int(row[0])-1]
        no_dict = now_dict.get(int(row[0]), {})
        n_dict = no_dict.get(int(row[2]), {})
        n_dict[int(row[1])] = row[:6]
        no_dict[int(row[2])] = n_dict
        now_dict[int(row[0])] = no_dict

    return now_dict
