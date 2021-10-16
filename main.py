import enum
import time
from io import DEFAULT_BUFFER_SIZE
import logging
import os
import random
from numpy.compat.py3k import is_pathlib_path
from numpy.lib.shape_base import _make_along_axis_idx
import torch
import shutil
import argparse
import itertools
import numpy as np

from loguru import logger
from datetime import datetime
from tqdm import tqdm

from torch import optim
from torch.nn import CrossEntropyLoss
from feeder_ngsim import NgsimFeeder, NgsimFeederII, NgsimFeederIII
from models import GRIP, TPHGI
from layers.graph import HeteroGraph, Graph
from utils import create_folders_if_not_exist



parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_test', action='store_true')
parser.add_argument('--resume_train_model_path', default='')
parser.add_argument('--model_name', default='TPHGI', type=str, choices=['GRIP', 'TPHGI'], help='GRIP or TPHGI')
parser.add_argument('--multi_task_with_lane', default=False, type=bool, help='Whether to do multi-task training.')
parser.add_argument('--num_hetero_types', default=3, type=int)
parser.add_argument('--seq2seq_type', default='gru', type=str, choices=['gru', 'lstm', 'transformer'])
parser.add_argument('--interact_in_decoding', default=True, type=bool)
parser.add_argument('--test_mode', default='all', type=str, choices=['all', 'compare'])
parser.add_argument('--train_batch_size', default=16, type=int, help='')
parser.add_argument('--val_batch_size', default=16, type=int, help='')
parser.add_argument('--test_batch_size', default=16, type=int, help='')
parser.add_argument('--num_epochs', default=500, type=int, help='')
parser.add_argument('--base_learning_rate', default=0.001, type=float, help='')
parser.add_argument('--lr_decay_epoch', default=1, type=int, help='')

parser.add_argument('--t_h', default=30, type=int, help='length of track history, seconds * sampling rate')
parser.add_argument('--t_f', default=50, type=int, help='length of predicted trajectory, seconds * sampling rate')
parser.add_argument('--down_sampling_steps', default=2, type=int)
parser.add_argument('--data_aug_ratio', default=0.5, type=float, help="data augmentation")
parser.add_argument('--neighbor_distance', default=90, type=float, help="it's unit is the feet")
parser.add_argument('--max_num_object', default=39, type=int)
args = parser.parse_args()

def seed_torch(seed=24):

	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def save_ckpt(state, checkpoint_dir, best_model_dir, is_best=False,  file_name='checkpoint.pt'):
    r"""Usage:
    >>> checkpoint = {
    >>>     'epoch': epoch + 1,
    >>>     'state_dict': model.state_dict(),
    >>>     'optimizer': optimizer.state_dict()
    >>> }
    >>> save_ckpt(checkpoint, checkpoint_dir, best_model_dir, is_best)
    """
    f_path = os.path.join(checkpoint_dir, file_name)
    torch.save(state, f_path)
    if is_best:
        best_f_path = os.path.join(best_model_dir, file_name)
        shutil.copyfile(f_path, best_f_path)

def load_ckpt(checkpoint_fpath, model, optimizer=None, lr_scheduler=None):
    r"""Usage:
    >>> model = MyModel(**kwargs)
    >>> optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    >>> ckpt_path = "path/to/checkpoint/checkpoint.pt"
    >>> model, optimizer, start_epoch = load_ckpt(ckpt_path, model, optimizer) 
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    outputs = (model, epoch)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        outputs += (optimizer, )
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        outputs += (lr_scheduler, )

    return outputs

def data_loader(data_fpath, batch_size=128, shuffle=True, drop_last=False, train_val_test='train'):

    dataset = NgsimFeederIII(data_fpath, train_val_test=train_val_test, **args.__dict__)
    dataset.length = 10000 # the length of val_dataset 12089
    # if train_val_test != 'train':
    #     dataset.length = 3000
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=True
    )
    return loader, dataset.length // batch_size + 1

def preprocess_data(data, rescale_xy, neighbor_matrices, observed_last):
    # data: (N, C, T, V)
    vehicle_ids = data[:, 1:2, :, :] # (N, 1, T, V)
    feature_id = [3, 4, 5, 6] # local x, local y, lane_id, mark
    ori_data = data[:, feature_id].detach()
    data = ori_data.detach().clone()

    new_mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0) # (N, 2, T-1, V)
    new_mask[:, :2, observed_last-1: observed_last+1, 0] = 1 # the first is the target vehicle and the last observed and the first predicted frames' mask of this vehicle must be 1. 
    # It is easier to predict the velocity of an object than predicting its location.
    # Calculate velocities p^{t+1} - p^{t} before feeding the data into the model.
    data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
    data[:, :2, 0] = 0

    data = data.float().to(args.device)
    data[:, :2] = data[:, :2] / rescale_xy
    ori_data = ori_data.float().to(args.device)
    vehicle_ids = vehicle_ids.long().to(args.device)

    adjacency_matrices = args.graph.get_adjacency_batch(neighbor_matrices)
    A = args.graph.normalize_adjacency_batch(adjacency_matrices).to(args.device)

    return data, ori_data, vehicle_ids, A

def display_RMSE_test(results, pref):
    all_overall_sum_list, all_overall_num_list = results
    overall_sum_time = np.sum(all_overall_sum_list, axis=0)
    # overall_sum_time = np.sum(all_overall_sum_list, axis=0)
    overall_num_time = np.sum(all_overall_num_list, axis=0)
    overall_loss_time = ((overall_sum_time / overall_num_time) ** 0.5) * 0.3048 # Calculate RMSE and convert from feet to meters
    overall_log = '|[{}] All_All: {}'.format(pref, ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))
    logger.info(overall_log)
    return overall_loss_time

def display_RMSE(pra_results, pra_pref='Train_epoch'):
    all_overall_sum_list, all_overall_num_list = pra_results
    overall_sum_time = np.sum(all_overall_sum_list**0.5, axis=0)
    # overall_sum_time = np.sum(all_overall_sum_list, axis=0)
    overall_num_time = np.sum(all_overall_num_list, axis=0)
    overall_loss_time = (overall_sum_time / overall_num_time) * 0.3048 # Calculate RMSE and convert from feet to meters
    overall_log = '|[{}] All_All: {}'.format(pra_pref, ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))
    logger.info(overall_log)
    return overall_loss_time

def compute_RMSE(pred, GT, mask, error_order=2):

    pred = pred * mask # (N, C, T, V)=(N, 2, 25, 120)
    GT = GT * mask # (N, C, T, V)=(N, 2, 25, 120)

    x2y2 = torch.sum(torch.abs(pred - GT) ** error_order, dim=1) # x^2+y^2, (N, C, T, V)->(N, T, V)=(N, 25, 255)
    overall_sum_time = x2y2.sum(dim=-1) # (N, T, V) -> (N, T)=(N, 25)
    overall_mask = mask.sum(dim=1).sum(dim=-1) # (N, C, T, V) -> (N, T)=(N, 25)
    overall_num = overall_mask

    return overall_sum_time, overall_num, x2y2


def train_model(model, data_loader, val_loader, optimizer, lr_scheduler, epoch_log, teacher_forcing_rate=0.0):
    model.train()

    rescale_xy = torch.ones((1,2,1,1)).to(args.device)
    rescale_xy[:, 0] = args.max_x
    rescale_xy[:, 1] = args.max_y
    class_criterion = CrossEntropyLoss(ignore_index=0) # we let 0 represent the unknown lane id
    for iteration, (ori_data, A, _, _) in tqdm(enumerate(data_loader)):
        # ori_data: (N, C, T, V), target_vehicle_ids: (N,)
        # C = 7 : [dataset_id, vehicle_id, frame_id local_x, local_y, lane_id] + [mark]
        data, _, _ = preprocess_data(ori_data, rescale_xy)
        #TODO change the range, decrease the training overload
        #for now_history_frames in range(1, data.shape[-2]): 
        optimizer.zero_grad()
        now_history_frames = args.t_h // args.down_sampling_steps + 1
        
        input_data = data[:, :, :now_history_frames, :]
        output_loc_GT = data[:, :2, now_history_frames:, :]
        output_mask = data[:, -1:, now_history_frames:, :] # (N, 1, T, V)
        lane_id_GT = data[:, 2:-1, now_history_frames:, :].detach().clone().long() # (N, 1, T, V)
        lane_id_GT = model.reshape_for_lstm(lane_id_GT) # (N*V, T, C), C = 1

        A = A.float().to(args.device)
        predicted, lane_id_predicted = model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=teacher_forcing_rate, pra_teacher_location=output_loc_GT)
        # Compute loss for training
        #TODO We use abs to compute loss to backward updata weights
        overall_sum_time, overall_num, _ = compute_RMSE(predicted, output_loc_GT, output_mask, error_order=1)
        # overall_loss
        total_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num), torch.ones(1,).to(args.device)) #(1,)
        if args.multi_task_with_lane:
            num_lane_id = lane_id_predicted.size(-1)
            lane_predict_loss = class_criterion(lane_id_predicted.view(-1, num_lane_id), lane_id_GT.view(-1))
            total_loss += lane_predict_loss
        
        
        if iteration % 30 == 0:
            now_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
            logger.info('|{}|{:>20}|\tIteration:{:>5}|\tLoss:{:.8f}|\tlr: {}|\ttf: {}|'.format(datetime.now(), epoch_log, iteration, total_loss.data.item(), now_lr, teacher_forcing_rate))            
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        lr_scheduler.step(total_loss.item())
        optimizer.zero_grad()


def val_model(model, data_loader, num_batch):
    model.eval()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(args.device)
    rescale_xy[:, 0] = args.max_x
    rescale_xy[:, 1] = args.max_y
    all_overall_sum_list = []
    all_overall_num_list = []
    with torch.no_grad():
        for iteration, (ori_data, neighbor_matrices, _, _) in tqdm(enumerate(data_loader), total=num_batch):
            now_history_frames = args.t_h // args.down_sampling_steps + 1 # 30 // 2 + 1, 30: history frames, 2: down sampling steps
            data, no_norm_loc_data, _, A = preprocess_data(ori_data, rescale_xy, neighbor_matrices, observed_last=now_history_frames-1)
            
            input_data = data[:, :, :now_history_frames, :]
            output_loc_GT = data[:, :2, now_history_frames:, :]
            output_mask = data[:, -1:, now_history_frames:, :]

            ori_output_loc_GT = no_norm_loc_data[:, :2, now_history_frames:, :]
            ori_output_last_loc = no_norm_loc_data[:, :2, now_history_frames-1: now_history_frames, :]

            A = A.float().to(args.device)
            predicted_velocity, _ = model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT)
            predicted = predicted_velocity * rescale_xy

            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind-1:ind+1], dim=-2)
            predicted += ori_output_last_loc

            
            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask, error_order=2)
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())

            # if len(all_overall_sum_list) > 945:
            #     print('stop here.')

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)

    #result = display_result([all_overall_sum_list, all_overall_num_list], pra_pref='Epoch:{}'.format(epoch))

    return all_overall_sum_list, all_overall_num_list

def test_model(model, data_loader, num_batch_test, mode='all'):
    model.eval()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(args.device)
    rescale_xy[:, 0] = args.max_x
    rescale_xy[:, 1] = args.max_y
    all_overall_sum_list = []
    all_overall_num_list = []
    with torch.no_grad():
        for iteration, (ori_data, neighbor_matrices, _, target_vehicle_ids) in tqdm(enumerate(data_loader), total=num_batch_test):
            now_history_frames = args.t_h // args.down_sampling_steps + 1 # 30 // 2 + 1, 30: history frames, 2: down sampling steps
            data, no_norm_loc_data, ori_vehicle_ids, A = preprocess_data(ori_data, rescale_xy, neighbor_matrices, observed_last=now_history_frames-1)
            input_data = data[:, :, :now_history_frames, :]
            output_loc_GT = data[:, :2, now_history_frames:, :]
            output_mask = data[:, -1:, now_history_frames:, :]

            ori_output_loc_GT = no_norm_loc_data[:, :2, now_history_frames:, :]
            ori_output_last_loc = no_norm_loc_data[:, :2, now_history_frames-1: now_history_frames, :]

            # A = A.float().to(args.device)
            predicted, _ = model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT)
            predicted = predicted * rescale_xy

            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind-1:ind+1], dim=-2)
            predicted += ori_output_last_loc

            if mode == 'compare': # compute the loss of target vehicles only.
                ori_vehicle_ids = ori_vehicle_ids[:, :, now_history_frames:, :]
                target_vehicle_ids = target_vehicle_ids.long().to(args.device)
                target_vehicle_ids = target_vehicle_ids.view(-1, 1, 1, 1).expand_as(ori_vehicle_ids) # (N, 1, T, V)
                target_vehicle_mask = ori_vehicle_ids == target_vehicle_ids
                output_mask *= target_vehicle_mask

            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask, error_order=2)
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)

    #result = display_result([all_overall_sum_list, all_overall_num_list], pra_pref='Epoch:{}'.format(epoch))

    return all_overall_sum_list, all_overall_num_list


def run_train_val(model, train_data_path, dev_data_path, resume_training=False):
    
    train_loader, num_batch_train = data_loader(
        train_data_path,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        train_val_test='train'
    )

    val_loader, num_batch_val = data_loader(
        dev_data_path,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        train_val_test='val'
    )

    optimizer = optim.AdamW(
        [
            {'params': model.parameters(), 'initial_lr': 0.0001}
        ],
        lr=args.base_learning_rate
    )

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.98, patience=100, min_lr=1e-5
    ) # (optimizer, step_size=100, gamma=0.99, last_epoch=40)
    trained_epochs = 0
    if resume_training:
        model, trained_epochs, optimizer, lr_scheduler = load_ckpt(args.resume_train_model_path, model, optimizer, lr_scheduler)
    
    model.to(args.device)
    mini_rmse = np.inf
    is_best = False
    num_iter_to_val = num_batch_train // 1.1
    for i_epoch in range(trained_epochs, args.num_epochs):

        logger.info('################# Train ##########################')
        teacher_forcing = np.exp(-i_epoch / 3)
        logger.info(f'current teacher forcing rate is {teacher_forcing}')

        # train_model(model, train_loader, val_loader, optimizer=optimizer, lr_scheduler=lr_scheduler, epoch_log='Epoch:{:>4}/{:>4}'.format(i_epoch, args.num_epochs), teacher_forcing_rate=teacher_forcing)
        model.train()

        rescale_xy = torch.ones((1,2,1,1)).to(args.device)
        rescale_xy[:, 0] = args.max_x
        rescale_xy[:, 1] = args.max_y
        class_criterion = CrossEntropyLoss(ignore_index=0) # we let 0 represent the unknown lane id
        for iteration, (ori_data, neighbor_matrices, _, target_vehicle_ids) in enumerate(train_loader):
            # ori_data: (N, C, T, V), target_vehicle_ids: (N,)
            # C = 7 : [dataset_id, vehicle_id, frame_id local_x, local_y, lane_id] + [mark]
            now_history_frames = args.t_h // args.down_sampling_steps + 1 # 30 // 2 + 1, 30: history frames, 2: down sampling steps
            data, no_norm_loc_data, _, A = preprocess_data(ori_data, rescale_xy, neighbor_matrices, observed_last=now_history_frames-1)
            # data, _, _, A = preprocess_data(ori_data, rescale_xy, neighbor_matrices)
            #TODO change the range, decrease the training overload
            #for now_history_frames in range(1, data.shape[-2]): 
            
            input_data = data[:, :, :now_history_frames, :]
            output_loc_GT = data[:, :2, now_history_frames:, :]
            output_mask = data[:, -1:, now_history_frames:, :] # (N, 1, T, V)
            lane_id_GT = data[:, 2:-1, now_history_frames:, :].detach().clone().long() # (N, 1, T, V)
            lane_id_GT = model.reshape_for_lstm(lane_id_GT) # (N*V, T, C), C = 1

            A = A.float().to(args.device)
            predicted, lane_id_predicted = model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=teacher_forcing, pra_teacher_location=output_loc_GT)
            # Compute loss for training
            #TODO We use abs to compute loss to backward updata weights
            overall_sum_time, overall_num, _ = compute_RMSE(predicted, output_loc_GT, output_mask, error_order=1)
            # overall_loss
            total_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num), torch.ones(1,).to(args.device)) #(1,)
            if args.multi_task_with_lane:
                num_lane_id = lane_id_predicted.size(-1)
                lane_predict_loss = class_criterion(lane_id_predicted.view(-1, num_lane_id), lane_id_GT.view(-1))
                total_loss += lane_predict_loss
            
            #DEBUG
            # i_b, i_t, i_v = torch.where(input_data[:, 1, :, :] > 50)
            # if len(i_b) > 0:
            #     print('stop here.')

            if iteration % 30 == 0:
                now_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                epoch_log='Epoch:{:>4}/{:>4}'.format(i_epoch, args.num_epochs)
                logger.info('|{:>20}|\tIteration:{:>5}|\tLoss:{:.6f}|\tlr: {:.5f}|\ttf: {:.3f}|'.format(epoch_log, iteration, total_loss.data.item(), now_lr, teacher_forcing))            
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_scheduler.step(total_loss.item())
            optimizer.zero_grad()
        
            if iteration % num_iter_to_val == 0 and iteration > 0:

                logger.info('################# Val ##########################')
                # s_t = time.time()
                rmse = display_RMSE_test(
                    val_model(model, val_loader, num_batch_val),
                    '{}_Epoch{}_Iteration{}'.format('Val', i_epoch, iteration)
                )
                if np.random.random() < 0.3:
                    display_RMSE_test(
                        test_model(model, val_loader, num_batch_val, mode='compare'),
                        '{}'.format('Val compare')
                    )
                # e_t = time.time()
                # logger.info(f'one training epoch and a validation process need {(e_t - s_t) / 60} minutes.')
                rmse = np.sum(rmse)
                if rmse < mini_rmse:
                    is_best = True
                    mini_rmse = rmse

                checkpoint = {
                    'epoch': i_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()
                }
                
                model_fname = f"{args.identity}_{i_epoch}_{iteration}.pt"
                save_ckpt(checkpoint, args.model_save_dir, args.best_model_save_dir, is_best=is_best, file_name=model_fname)
                is_best = False
                model.train()


def run_test(model, test_data_path, model_path=None, mode='all'):
    if model_path:
        model, _, _ = load_ckpt(model_path, model)
    model.to(args.device)
    logger.info(f'################# Test with mode: {mode} ##########################')
    test_loader, num_batch_test = data_loader(
        test_data_path,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        train_val_test='test'
    )
    display_RMSE_test(
        test_model(model, test_loader, num_batch_test, mode=mode),
        '{}'.format('Test')
    )


if __name__ == '__main__':

    # Set random seed for reproducibility.
    seed_torch()

    # Create the output folder.
    output_root = 'outputs'
    model_save_dir = os.path.join(output_root, 'saved_models')
    best_model_save_dir = os.path.join(model_save_dir, 'best_models')
    log_dir = os.path.join(output_root, 'logs')
    create_folders_if_not_exist(model_save_dir, log_dir, best_model_save_dir)

    
    args.model_save_dir = model_save_dir
    args.best_model_save_dir = best_model_save_dir
    args.log_dir = log_dir
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_path = 'data/ValSet.mat' #TODO 仅用于测试
    dev_path = 'data/ValSet.mat'
    test_path = 'data/TestSet.mat'

    args.graph_args = {'max_hop': 2, 'num_node': args.max_num_object,  'num_hetero_types': args.num_hetero_types}
    args.use_hetero_graph = True if args.model_name == 'TPHGI' else False
    if args.use_hetero_graph:
        args.graph = HeteroGraph(**args.graph_args)
    else:
        args.graph = Graph(**args.graph_args)
        args.num_hetero_types = 0
    # args.multi_task_with_lane = True
    assert not args.multi_task_with_lane if args.model_name == 'GRIP' else True

    # Set logger
    args.identity = f'{args.model_name}_{args.seq2seq_type}_{int(args.num_hetero_types)}_{int(args.interact_in_decoding)}' + \
                    f'_{int(args.multi_task_with_lane)}_{args.base_learning_rate}_{args.data_aug_ratio}'
    log_file = os.path.join(log_dir,  f'{args.identity}_.log')
    logger.add(log_file, format="<green>{time}</green> <level>{message}</level>", level="INFO")

    model_dict = {
        'GRIP': GRIP,
        'TPHGI': TPHGI
    }
    model = model_dict[args.model_name](
        in_channels=4, graph_args=args.graph_args, edge_importance_weighting=True, max_num_object=args.max_num_object,
        predict_lane=args.multi_task_with_lane, seq2seq_type=args.seq2seq_type, interact_in_decoding=args.interact_in_decoding
    )
    args.max_x = 1.0 # 0.1 # 5.2
    args.max_y = 1.0 # 1.0 # 32
    args.do_train = True
    if args.do_train:
        resume_training = len(args.resume_train_model_path) > 0 # continue training or not
        run_train_val(model, train_data_path=train_path, dev_data_path=dev_path, resume_training=resume_training)
    if args.do_test:
        model_path = 'outputs/saved_models/best_models/GRIP_gru_0_0_0.001_0.0_3.pt'
        run_test(model, test_data_path=dev_path, model_path=model_path, mode='all')
        run_test(model, test_data_path=dev_path, model_path=model_path, mode='compare')



