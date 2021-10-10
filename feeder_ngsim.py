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
        frame_dict = defaultdict(lambda: dict())
        for vehicle_idx, vehicle_data in enumerate(dataset):
            if len(vehicle_data) <= 1:
                continue
            assert vehicle_data.shape[0] == 4
            vehicle_id = vehicle_idx + 1
            for frame_id, local_x, local_y, lane_id in vehicle_data.transpose():
                frame_dict[int(frame_id)][int(vehicle_id)] = np.array([dataset_id, vehicle_id, frame_id, local_x, local_y, lane_id], dtype=np.float32)
        dataset_list.append(frame_dict)
        sorted_frame_id_list.append(sorted(frame_dict.keys()))
    return dataset_list, sorted_frame_id_list


class NgsimFeederII(Dataset):
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
        if self.use_hetero_graph:
            
            self.graph = HeteroGraph(**kwargs.get('graph_args', graph_args))
            self.num_hetero_types = self.graph.num_hetero_types
            
        else:
            self.graph = Graph(**kwargs.get('graph_args', graph_args))
        self._load_and_process_data()

        # num_datasets = len(self.dataset_dict)
        # self.num_examples_dataset = OrderedDict() # the number of examples in each dataset
        

    def _load_and_process_data(self):
        
        # load data from .mat format file.
        mat_data = scp.loadmat(self.data_path)
        # self.traj: shape of (*, 47), 1: Dataset Id 2: Vehicle Id 3: Frame Id 4: Local X 5: Local Y 6: Lane Id 7: Lateral maneuver 8: Longitudinal maneuver 9-47: Neighbor Car Ids at grid location
        # self.tracks: shape of (num_dataset, *), where num_dataset = 6
        self.traj, tracks = mat_data['traj'], mat_data['tracks']
        self.length = len(self.traj)
        self.dataset_list, self.sorted_frame_id_set_dataset = get_frame_instance_dict_new(tracks)
        # self.dataset_dict = get_frame_instance_dict(self.traj)
        
    def __len__(self):
        
        return self.length
    
    def __getitem__(self, index):

        cur_dataset_id, vehicle_id, frame_id = self.traj[index][:3]
        cur_dataset_id, frame_id = int(cur_dataset_id), int(frame_id)
        # cur_dataset_id, cur_frame_idx = self._get_current_dataset_frame_id(index)
        sorted_frame_id_set = self.sorted_frame_id_set_dataset[cur_dataset_id-1]
        cur_frame_idx = bisect_left(sorted_frame_id_set, frame_id)
        now_dict = self.dataset_list[cur_dataset_id-1]
        object_frame_feature, neighbor_matrix, mean_xy = self._process_data(now_dict, cur_frame_idx, sorted_frame_id_set)
        object_frame_feature = object_frame_feature.transpose(2, 1, 0)
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

        adjacency_maxtrix = self.graph.get_adjacency(neighbor_matrix)
        A = self.graph.normalize_adjacency(adjacency_maxtrix)


        return object_frame_feature, A, mean_xy, int(vehicle_id)

    def _process_data(self, now_dict, cur_frame_idx, sorted_frame_id_set):

        start_ind, end_ind = cur_frame_idx - self.history_frames, cur_frame_idx + self.future_frames + 1
        # observed_last = list(range(start_ind, end_ind, self.down_sampling_steps))[self.history_frames // self.down_sampling_steps - 1]
        observed_last = cur_frame_idx
        observed_last_frame = sorted_frame_id_set[observed_last]
        observed_last_frame_feature = now_dict[observed_last_frame] # {vehicle_id: features}
        # start_ind, end_ind = max(0, cur_frame_idx - self.history_frames + 1), min(len(sorted_frame_id_set), cur_frame_idx + self.future_frames + 1)
        visible_object_id_set = set(observed_last_frame_feature.keys())
        num_visible_object = len(visible_object_id_set)
        
        # compute the mean values of x and y for zero-centralization
        visible_object_value = np.array(list(observed_last_frame_feature.values())) # we only use the first six features: [dataset_id, vehicle_id, frame_id, local_x, local_y, lane_id]
        xy = visible_object_value[:, 3:5].astype(float)
        mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
        m_xy = np.mean(xy, axis=0)
        mean_xy[3:5] = m_xy
        
        # object_id appears in the range of start_ind to end_ind frames.
        now_all_object_id_set = set([val for x in sorted_frame_id_set[start_ind: end_ind] for val in now_dict[x].keys()])
        num_non_visible_object = len(now_all_object_id_set) - num_visible_object
        total_feature_dimension = visible_object_value.shape[-1] + 1 # will add a "mark" feature.
        
        # for any pair of two objects, we think they are neighbors if their distance in the logitudinal direction is less than neighbor_distance
        # and within the two adjacent lanes, this information has been recorded in self.D
        neighbor_matrix = np.zeros((self.max_num_object, self.max_num_object))
        neighbor_matrix_visible = np.zeros((num_visible_object, num_visible_object), dtype=np.int8)
        for i in range(num_visible_object):
            for j in range(0, i+1):
                longitu_i, lane_i = visible_object_value[i][4:6]
                longitu_j, lane_j = visible_object_value[j][4:6]
                if abs(longitu_i - longitu_j) < self.neighbor_distance and abs(lane_i - lane_j) < 2.0:
                    neighbor_matrix_visible[i][j] = neighbor_matrix_visible[j][i] = 1

        if self.use_hetero_graph:
            neighbor_matrix_visible = self._fill_hetero_type(neighbor_matrix_visible, visible_object_value)
        neighbor_matrix[:num_visible_object, :num_visible_object] = neighbor_matrix_visible

        # for all history frames() or feature frames, we only choose the objects listed in visible_object_id_set
        object_feature_list = []
        # down sampling
        # assert sorted_frame_id_set[cur_frame_idx] - sorted_frame_id_set[start_ind] == cur_frame_idx - start_ind
        # assert sorted_frame_id_set[end_ind-1] - sorted_frame_id_set[cur_frame_idx] == end_ind-1 - cur_frame_idx
        frame_idxs_after_down_sampling = list(range(start_ind, end_ind, self.down_sampling_steps))[1:]
        assert observed_last in frame_idxs_after_down_sampling
        start_frame_id = sorted_frame_id_set[start_ind]
        for idx in frame_idxs_after_down_sampling:
            frame_id = sorted_frame_id_set[idx]
            # we add mark "1" to the end of each row to indicate that this row exists, using list(now_dict[frame_id][obj_id]) + [1]
            # besides, we need to confirm the frame numbers are continous. 
            # -mean_xy is used to zero_centralize data
            now_frame_feature_dict = {
                obj_id: (list(now_dict[frame_id][obj_id] - mean_xy) + [1]
                if obj_id in visible_object_id_set else list(now_dict[frame_id][obj_id] - mean_xy) + [0]) #  and frame_id-start_frame_id == idx-start_ind
                for obj_id in now_dict[frame_id]
            }
            # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(7))
            now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in now_all_object_id_set])
            object_feature_list.append(now_frame_feature)
        
        # object_feature_list has shape of (#frame, #object, 7), 7 = 6 features + 1 mark
        object_feature_list = np.array(object_feature_list)

        # object_frame_feature with a shape of (#frame, #object, 7) -> (#object, #frame, 7)
        # num_frames_after_down_sampling = end_ind - start_ind - ((observed_last + 1 - start_ind) // 2)
        object_frame_feature = np.zeros((self.max_num_object, len(frame_idxs_after_down_sampling), total_feature_dimension))
        
        object_frame_feature[:num_visible_object + num_non_visible_object, :len(object_frame_feature)] = np.transpose(object_feature_list, (1,0,2))
        
        return object_frame_feature, neighbor_matrix, m_xy


    def _fill_hetero_type(self, neighbor_matrix, visible_object_value):
        neighbor_type_matrix = np.zeros_like(neighbor_matrix)
        for i in range(neighbor_matrix.shape[0]):
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
    

    def _get_current_dataset_frame_id(self, index):

        new_index = index
        for dataset_id, num_examples in self.num_examples_dataset.items():
            if new_index < num_examples:
                return dataset_id, new_index
            else:
                new_index -= num_examples



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
        if self.use_hetero_graph:
            
            self.graph = HeteroGraph(**kwargs.get('graph_args', graph_args))
            self.num_hetero_types = self.graph.num_hetero_types
            
        else:
            self.graph = Graph(**kwargs.get('graph_args', graph_args))
        self.load_and_process_data()

        self.length = 0
        num_datasets = len(self.dataset_dict)
        self.num_examples_dataset = OrderedDict() # the number of examples in each dataset
        self.sorted_frame_id_set_dataset = dict() # the set of frame id for each dataset
        for dataset_id, data_dict in self.dataset_dict.items():
            frame_id_set = sorted(set(data_dict.keys())) # key: dataset id
            num_examples = len(frame_id_set) - self.total_frames + 1
            self.length += num_examples
            self.num_examples_dataset[dataset_id] = num_examples
            self.sorted_frame_id_set_dataset[dataset_id] = frame_id_set


    def load_and_process_data(self):
        
        # load data from .mat format file.
        mat_data = scp.loadmat(self.data_path)
        # self.traj: shape of (*, 47), 1: Dataset Id 2: Vehicle Id 3: Frame Id 4: Local X 5: Local Y 6: Lane Id 7: Lateral maneuver 8: Longitudinal maneuver 9-47: Neighbor Car Ids at grid location
        # self.tracks: shape of (num_dataset, *), where num_dataset = 6
        self.traj, self.tracks = mat_data['traj'], mat_data['tracks']

        self.dataset_dict = get_frame_instance_dict(self.traj)
        

    def __len__(self):
        
        return self.length
    
    def __getitem__(self, index):
        cur_dataset_id = None
        cur_frame_number = None
        new_index = index
        for dataset_id, num_examples in self.num_examples_dataset.items():
            if new_index < num_examples:
                cur_dataset_id, cur_frame_number = dataset_id, new_index
                break
            else:
                new_index -= num_examples

        now_dict = self.dataset_dict[cur_dataset_id]
        sorted_frame_id_set = self.sorted_frame_id_set_dataset[cur_dataset_id]
        start_ind, end_ind = cur_frame_number, cur_frame_number + self.total_frames
        observed_last = cur_frame_number + self.history_frames - 1
        object_frame_feature, neighbor_matrix, mean_xy = self.process_data(
            now_dict, start_ind, end_ind, observed_last, sorted_frame_id=sorted_frame_id_set
        )
        # (V, T, C) -> (C, T, V)
        object_frame_feature = object_frame_feature.transpose(2, 1, 0)

        # data augmentation rate = 0.5
        if self.train_val_test.lower() == 'train' and np.random.random() < self.data_aug_ratio:
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

        adjacency_maxtrix = self.graph.get_adjacency(neighbor_matrix)
        A = self.graph.normalize_adjacency(adjacency_maxtrix)


        return object_frame_feature, A, mean_xy

    def process_data(self, now_dict, start_ind, end_ind, observed_last, sorted_frame_id):
        """
        """
        real_observed_last = sorted_frame_id[observed_last]

        visible_object_id_set = set(now_dict[real_observed_last].keys()) # object_id appears at the last observed frame
        num_visible_object = len(visible_object_id_set)

        # compute the mean values of x and y for zero-centralization.
        visible_object_value = np.array(list(now_dict[real_observed_last].values()))
        xy = visible_object_value[:, 3:5].astype(float)
        mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
        m_xy = np.mean(xy, axis=0)
        mean_xy[3:5] = m_xy

        # object_id appears in the range of start_ind to end_ind frames.
        # TODO for simplicity, we change the range (start_ind, end_ind) to (observed_last, end_ind)
        now_all_object_id = set([val for x in sorted_frame_id[start_ind:end_ind] for val in now_dict[x].keys()])
        non_visible_object_id_set = now_all_object_id - visible_object_id_set
        num_non_visible_object = len(non_visible_object_id_set)
        if len(now_all_object_id) > self.num_object_record:
            self.num_object_record = len(now_all_object_id)
            print(f"Maximum number of vehicles: {self.num_object_record}")

        total_feature_dimension = visible_object_value.shape[-1] + 1 # will add a "mark" feature.
        # compute distance between any pair of two objects
        # dist_xy = spatial.distance.cdist(xy, xy)
        # neighbor_matrix_partial = (dist_xy < self.neighbor_distance).astype(int)
        # for any pair of two objects, we think they are neighbors if their distance in the logitudinal direction is less than neighbor_distance and within the two adjacent lanes
        neighbor_matrix = np.zeros((self.max_num_object, self.max_num_object))
        neighbor_matrix_visible = np.zeros((num_visible_object, num_visible_object), dtype=np.int8)
        for i in range(num_visible_object):
            for j in range(num_visible_object):
                longitu_i, lane_i = visible_object_value[i][4:6]
                longitu_j, lane_j = visible_object_value[j][4:6]
                if abs(longitu_i - longitu_j) < self.neighbor_distance and abs(lane_i - lane_j) < 2.0:
                    neighbor_matrix_visible[i][j] = neighbor_matrix_visible[j][i] = 1
        if self.use_hetero_graph:
            neighbor_matrix_visible = self.fill_hetero_type(neighbor_matrix_visible, visible_object_value)
        neighbor_matrix[:num_visible_object, :num_visible_object] = neighbor_matrix_visible

        # for all history frames() or feature frames, we only choose the objects listed in visible_object_id_set
        object_feature_list = []
        # down sampling
        assert sorted_frame_id[observed_last] - sorted_frame_id[start_ind] == observed_last - start_ind
        assert sorted_frame_id[end_ind-1] - sorted_frame_id[observed_last] == end_ind-1 - observed_last
        frames_after_down_sampling = sorted_frame_id[start_ind: end_ind: self.down_sampling_steps]
        
        for frame_ind in frames_after_down_sampling:
            
            # we add mark "1" to the end of each row to indicate that this row exists, using list(now_dict[frame_id][obj_id]) + [1]
            # -mean_xy is used to zero_centralize data
            now_frame_feature_dict = {
                obj_id: (list(now_dict[frame_ind][obj_id] - mean_xy) + [1]
                if obj_id in visible_object_id_set else list(now_dict[frame_ind][obj_id] - mean_xy) + [0])
                for obj_id in now_dict[frame_ind]
            }
            # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(7))
            now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in now_all_object_id])
            object_feature_list.append(now_frame_feature)
        
        # object_feature_list has shape of (#frame, #object, 7), 7 = 6 features + 1 mark
        object_feature_list = np.array(object_feature_list)

        # object_frame_feature with a shape of (#frame, #object, 7) -> (#object, #frame, 7)
        # num_frames_after_down_sampling = end_ind - start_ind - ((observed_last + 1 - start_ind) // 2)
        object_frame_feature = np.zeros((self.max_num_object, len(frames_after_down_sampling), total_feature_dimension))
        assert num_visible_object + num_non_visible_object == len(now_all_object_id)
        object_frame_feature[:num_visible_object + num_non_visible_object] = np.transpose(object_feature_list, (1,0,2))
        
        return object_frame_feature, neighbor_matrix, m_xy

    def fill_hetero_type(self, neighbor_matrix, visible_object_value):
        neighbor_type_matrix = np.zeros_like(neighbor_matrix)
        for i in range(neighbor_matrix.shape[0]):
            for j in range(i):
                if neighbor_matrix[i][j] == 1:
                    lane_id_i = visible_object_value[i][5] # lane_id is the six-th feature in visible_object_value
                    lane_id_j = visible_object_value[j][5]
                    if lane_id_i > lane_id_j:
                        neighbor_type_matrix[i][j], neighbor_type_matrix[j][i] = self.num_hetero_types-1, self.num_hetero_types
                    elif lane_id_i < lane_id_j:
                        neighbor_type_matrix[i][j], neighbor_type_matrix[j][i] = self.num_hetero_types, self.num_hetero_types-1

        for i in range(neighbor_matrix.shape[0]):
            neighbor_type_matrix[i][i] = 1

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

if __name__ == '__main__':
    
    def preprocess_data(data, rescale_xy):
        # data: (N, C, T, V)
        feature_id = [3, 4, 5, 6] # local x, local y, lane_id, mark
        ori_data = data[:, feature_id].detach()
        data = ori_data.detach().clone()

        new_mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0) 
        # It is easier to predict the velocity of an object than predicting its location.
        # Calculate velocities p^{t+1} - p^{t} before feeding the data into the model.
        data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
        data[:, :2, 0] = 0

        data = data.float()
        data[:, :2] = data[:, :2] / rescale_xy
        ori_data = ori_data.float()

        return data, ori_data

    mat_fpath = 'data/ValSet.mat'
    dataset = NgsimFeederII(mat_fpath, train_val_test='val', max_num_object=255, **{'graph_args': {'max_hop': 2, 'num_node': 255, 'num_hetero_types': 3}})
    dataset.__getitem__(3)
    # dataset.length = 100
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
   
    
    rescale_xy = torch.ones((1,2,1,1))
    rescale_xy[:, 0] = 1.0
    rescale_xy[:, 1] = 1.0
    max_velocity_x, max_velocity_y = 0.0, 0.0
    for iteration, (ori_data, A, _) in tqdm(enumerate(loader)):
        
        data, no_norm_loc_data = preprocess_data(ori_data, rescale_xy)
        cur_max_x = data[:, 0, :, :].abs().max().item()
        cur_max_y = data[:, 1, :, :].abs().max().item()
        max_velocity_x = cur_max_x if cur_max_x > max_velocity_x else max_velocity_x
        max_velocity_y = cur_max_y if cur_max_y > max_velocity_y else max_velocity_y

    print(max_velocity_x, max_velocity_y)