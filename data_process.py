import os
import glob
import pickle
import numpy as np
import scipy.io as scp

from tqdm import tqdm


history_frames = 30
future_frames = 50

total_frames = history_frames + future_frames
max_num_object = 255
neighbor_distance = 90 # feet
down_sampling_steps = 2

use_hetero_graph = False


def get_frame_instance_dict(traj):
    """
    Read data from traj and return a dictionary:
    {dataset_id
        {frame_id:
            {object_id:
                # 6 features
                [dataset_id, vehicle_id, frame_id, local_x, local_y, lane_id]
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
        n_dict[int(row[1])] = row[:6]
        no_dict[int(row[2])] = n_dict
        now_dict[int(row[0])] = no_dict

    return now_dict


def process_data(now_dict, cur_frame_idx, sorted_frame_id_set):

    observed_last_frame = sorted_frame_id_set[cur_frame_idx]
    observed_last_frame_feature = now_dict[observed_last_frame] # {vehicle_id: features}
    start_ind, end_ind = max(0, cur_frame_idx - history_frames + 1), cur_frame_idx + future_frames + 1
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
    neighbor_matrix = np.zeros((max_num_object, max_num_object))
    neighbor_matrix_visible = np.zeros((num_visible_object, num_visible_object), dtype=np.int8)
    for i in range(num_visible_object):
        for j in range(0, i+1):
            longitu_i, lane_i = visible_object_value[i][4:6]
            longitu_j, lane_j = visible_object_value[j][4:6]
            if abs(longitu_i - longitu_j) < neighbor_distance and abs(lane_i - lane_j) < 2.0:
                neighbor_matrix_visible[i][j] = neighbor_matrix_visible[j][i] = 1

    if use_hetero_graph:
        neighbor_matrix_visible = fill_hetero_type(neighbor_matrix_visible, visible_object_value)
    neighbor_matrix[:num_visible_object, :num_visible_object] = neighbor_matrix_visible

    # for all history frames() or feature frames, we only choose the objects listed in visible_object_id_set
    object_feature_list = []
    # down sampling
    assert sorted_frame_id_set[cur_frame_idx] - sorted_frame_id_set[start_ind] == cur_frame_idx - start_ind
    assert sorted_frame_id_set[end_ind-1] - sorted_frame_id_set[cur_frame_idx] == end_ind-1 - cur_frame_idx
    frames_after_down_sampling = sorted_frame_id_set[start_ind: end_ind: down_sampling_steps]
    
    for frame_ind in frames_after_down_sampling:
        
        # we add mark "1" to the end of each row to indicate that this row exists, using list(now_dict[frame_id][obj_id]) + [1]
        # -mean_xy is used to zero_centralize data
        now_frame_feature_dict = {
            obj_id: (list(now_dict[frame_ind][obj_id] - mean_xy) + [1]
            if obj_id in visible_object_id_set else list(now_dict[frame_ind][obj_id] - mean_xy) + [0])
            for obj_id in now_dict[frame_ind]
        }
        # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(7))
        now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in now_all_object_id_set])
        object_feature_list.append(now_frame_feature)
    
    # object_feature_list has shape of (#frame, #object, 7), 7 = 6 features + 1 mark
    object_feature_list = np.array(object_feature_list)

    # object_frame_feature with a shape of (#frame, #object, 7) -> (#object, #frame, 7)
    # num_frames_after_down_sampling = end_ind - start_ind - ((observed_last + 1 - start_ind) // 2)
    object_frame_feature = np.zeros((max_num_object, total_frames // down_sampling_steps, total_feature_dimension))
    
    object_frame_feature[:num_visible_object + num_non_visible_object, :len(frames_after_down_sampling)] = np.transpose(object_feature_list, (1,0,2))
    
    return object_frame_feature, neighbor_matrix, m_xy



def fill_hetero_type(self, neighbor_matrix, visible_object_value):
    """
    in the same lane: 1
    in the left lane: 3
    in the right lane: 2
    """
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



def generate_data(mat_fpath, train_val_test='train'):

    mat_data = scp.loadmat(mat_fpath)
    # self.traj: shape of (*, 47), 1: Dataset Id 2: Vehicle Id 3: Frame Id 4: Local X 5: Local Y 6: Lane Id 7: Lateral maneuver 8: Longitudinal maneuver 9-47: Neighbor Car Ids at grid location
    # self.tracks: shape of (num_dataset, *), where num_dataset = 6
    traj, tracks = mat_data['traj'], mat_data['tracks']
    dataset_dict = get_frame_instance_dict(traj)
    
    for dataset_id, now_dict in tqdm(dataset_dict.items()):
        if int(dataset_id) != 2:
            continue
        feature_list = []
        adjacency_list = []
        mean_list = []
        sorted_frame_id_set = sorted(now_dict.keys())

        for cur_frame_idx in tqdm(range(len(sorted_frame_id_set) - future_frames)):

            object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, cur_frame_idx, sorted_frame_id_set)
            feature_list.append(object_frame_feature)
            adjacency_list.append(neighbor_matrix)
            mean_list.append(mean_xy)

    # (N, V, T, C) --> (N, C, T, V)
        all_feature = np.transpose(feature_list, (0, 3, 2, 1))
        all_adjacency = np.array(adjacency_list)
        all_mean = np.array(mean_list)
        print(all_feature.shape, all_adjacency.shape, all_mean.shape)

        save_path = f"data/{train_val_test}_{dataset_id}_data.pkl"
        with open(save_path, 'wb') as writer:
            pickle.dump([all_feature, all_adjacency, all_mean], writer, protocol=pickle.HIGHEST_PROTOCOL)
	
if __name__ == '__main__':

    train_mat_fpath = 'data/TrainSet.mat'
    dev_mat_path = 'data/ValSet.mat'
    test_mat_path = 'data/TestSet.mat'

    print('Generating Training Data.')
    generate_data(train_mat_fpath)

    print('Generating Validation Data.')
    generate_data(dev_mat_path, 'val')

    print('Generating Testing Data.')
    generate_data(test_mat_path, 'test')

