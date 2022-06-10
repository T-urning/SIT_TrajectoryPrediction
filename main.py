
import os

import torch
import argparse
import numpy as np

from loguru import logger
from tqdm import tqdm

from torch import optim
from torch.nn import CrossEntropyLoss
from feeder_ngsim import NgsimFeeder
from models import SpatialTransformerRNN
from utils import create_folders_if_not_exist, seed_torch, save_ckpt, load_ckpt



parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_test', action='store_true')
parser.add_argument('--predict_velocity', default=True, type=bool)
parser.add_argument('--resume_train_model_path', default='', help='Continue training from a given model state.')
parser.add_argument('--model_name', default='STransformerRNN', type=str, choices=['GRIP', 'TPHGI', 'STransformerRNN'])
parser.add_argument('--transformer_layers', default=4, type=int)
parser.add_argument('--seq2seq_type', default='gru', type=str, choices=['gru', 'lstm', 'transformer'])
parser.add_argument('--interact_in_decoding', default=False, type=bool, help='Whether to consider spatial interactions among vehicles in decoding/predicting.')
parser.add_argument('--spatial_interact', default=True, type=bool, help='Whether to consider spatial interactions in encoding.')
parser.add_argument('--teacher_forcing', default=0.6, type=float, help='Used in decoding when training.')
parser.add_argument('--test_mode', default='all', type=str, choices=['all', 'compare'])
parser.add_argument('--train_batch_size', default=32, type=int, help='')
parser.add_argument('--val_batch_size', default=32, type=int, help='')
parser.add_argument('--test_batch_size', default=64, type=int, help='')
parser.add_argument('--num_epochs', default=300, type=int, help='')
parser.add_argument('--base_learning_rate', default=0.0001, type=float, help='')
parser.add_argument('--lr_decay_epoch', default=1, type=int, help='')

parser.add_argument('--t_h', default=30, type=int, help='length of track history, seconds * sampling rate')
parser.add_argument('--t_f', default=50, type=int, help='length of predicted trajectory, seconds * sampling rate')
parser.add_argument('--down_sampling_steps', default=2, type=int)
parser.add_argument('--data_aug_ratio', default=0.0, type=float, help="data augmentation")
parser.add_argument('--neighbor_distance', default=50, type=float, help="it's unit is the feet")
parser.add_argument('--max_num_object', default=22, type=int)
args = parser.parse_args()



def data_loader(*data_fpaths, batch_size=128, shuffle=True, drop_last=False, train_val_test='train'):
    datasets = []
    for data_path in data_fpaths:
        dataset = NgsimFeeder(data_path, train_val_test=train_val_test, **args.__dict__)
        datasets.append(dataset)
    
    dataset = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=2,
        pin_memory=True
    )
    return loader, dataset.cummulative_sizes[-1] // batch_size + 1

def preprocess_data(data, rescale_xy, neighbor_matrices, observed_last):
    # data: (N, C, T, V)
    # vehicle_ids = data[:, 1:2, :, :] # (N, 1, T, V)
    feature_id = [3, 4, 5, 6] # local x, local y, lane_id, mark
    ori_data = data[:, feature_id].detach()
    data = ori_data.detach().clone()

    new_mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0) # (N, 2, T-1, V)
    new_mask[:, :2, observed_last-1: observed_last+1, 0] = 1 # the first is the target vehicle and the last observed and the first predicted frames' mask of this vehicle must be 1. 
    # It is easier to predict the velocity of an object than predicting its location.
    # Calculate velocities p^{t+1} - p^{t} before feeding the data into the model.
    if args.predict_velocity:
        data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
        data[:, :2, 0] = 0

    data = torch.cat([data[:, :2], ori_data[:, :2]], dim=1) # concat velocity and origin location.

    data = data.float().to(args.device)
    data[:, :4] = data[:, :4] / rescale_xy
    ori_data = ori_data.float().to(args.device)
    # vehicle_ids = vehicle_ids.long().to(args.device)

    A = neighbor_matrices.to(args.device)

    
    return data, ori_data, A
    

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


def val_model(model, data_loader, num_batch_test, congest_scene_only=False):
    model.eval()
    rescale_xy = torch.ones((1, 4, 1, 1)).to(args.device)
    rescale_xy[:, 0] = args.max_x_velocity
    rescale_xy[:, 1] = args.max_y_velocity
    rescale_xy[:, 2] = args.max_x
    rescale_xy[:, 3] = args.max_y
    all_overall_sum_list, compare_overall_sum_list = [], []
    all_overall_num_list, compare_overall_num_list = [], []
    total_scenes, congested_scenes = 0, 0
    with torch.no_grad():
        for iteration, (ori_data, neighbor_matrices, _, num_observed_vehicles) in tqdm(enumerate(data_loader), total=num_batch_test):
            now_history_frames = args.t_h // args.down_sampling_steps + 1 # 30 // 2 + 1, 30: history frames, 2: down sampling steps
            data, no_norm_loc_data, neighbor_matrices = preprocess_data(ori_data, rescale_xy, neighbor_matrices, observed_last=now_history_frames-1)
            input_data = data[:, :, :now_history_frames, :]
            output_loc_GT = data[:, :2, now_history_frames:, :]
            output_mask = no_norm_loc_data[:, -1:, now_history_frames:, :]
            input_mask = no_norm_loc_data[:, -1:, :now_history_frames, :]
            A = neighbor_matrices[:, :now_history_frames]

            ori_output_loc_GT = no_norm_loc_data[:, :2, now_history_frames:, :]
            ori_output_last_loc = no_norm_loc_data[:, :2, now_history_frames-1: now_history_frames, :]

            predicted, _ = model(pra_x=input_data, pra_A=A, input_mask=input_mask, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT)
            predicted = predicted * rescale_xy[:, :2, :, :]

            if args.predict_velocity:
                for ind in range(1, predicted.shape[-2]):
                    predicted[:, :, ind] = torch.sum(predicted[:, :, ind-1:ind+1], dim=-2)
                predicted += ori_output_last_loc

            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask, error_order=2)
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
            # if mode == 'compare': # compute the loss of target vehicles only.
            output_mask[:, :, :, 1:] = 0.0
            total_scenes += output_mask.size(0)
            if congest_scene_only:
                is_congested_scenes = num_observed_vehicles >= 10
                output_mask = output_mask[is_congested_scenes]
                predicted = predicted[is_congested_scenes]
                ori_output_loc_GT = ori_output_loc_GT[is_congested_scenes]
                congested_scenes += output_mask.size(0)
            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask, error_order=2)
            compare_overall_num_list.extend(overall_num.detach().cpu().numpy())
            compare_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())



    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)

    compare_overall_sum_list = np.array(compare_overall_sum_list)
    compare_overall_num_list = np.array(compare_overall_num_list)

    logger.info(f'total scenes: {total_scenes}, congested scenes: {congested_scenes}')
    #result = display_result([all_overall_sum_list, all_overall_num_list], pra_pref='Epoch:{}'.format(epoch))

    return all_overall_sum_list, all_overall_num_list, compare_overall_sum_list, compare_overall_num_list


def run_train_val(model, train_data_path, dev_data_path, test_data_path, resume_training=False):
    
    train_loader, num_batch_train = data_loader(
        train_data_path, # dev_data_path
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        train_val_test='train'
    )

    val_loader, num_batch_val = data_loader(
        dev_data_path,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        train_val_test='val'
    )

    # test_loader, num_batch_test = data_loader(
    #     test_data_path,
    #     batch_size=args.val_batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     train_val_test='test'
    # )


    model.to(args.device)

    optimizer = optim.AdamW(
        [
            {'params': model.parameters(), 'initial_lr': 0.0001}
        ],
        lr=args.base_learning_rate
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_batch_train, gamma=0.5)

    trained_epochs = 0
    if resume_training:
        logger.info(f'resume model training from {args.resume_train_model_path}')
        model, trained_epochs, optimizer, lr_scheduler = load_ckpt(
            args.resume_train_model_path, model, optimizer, lr_scheduler)
    
    
    mini_rmse = np.inf
    is_best = False
    num_iter_to_val = num_batch_train // 5
    teacher_forcing_ratio = [args.teacher_forcing, 0.2, 0.0]
    for i_epoch in range(trained_epochs, args.num_epochs):

        logger.info('################# Train ##########################')
        teacher_forcing = teacher_forcing_ratio[i_epoch] if i_epoch < len(teacher_forcing_ratio) else teacher_forcing_ratio[-1] # np.exp(-(i_epoch+1) / 1.5)
        logger.info(f'current teacher forcing rate is {teacher_forcing}')

        # train_model(model, train_loader, val_loader, optimizer=optimizer, lr_scheduler=lr_scheduler, epoch_log='Epoch:{:>4}/{:>4}'.format(i_epoch, args.num_epochs), teacher_forcing_rate=teacher_forcing)
        model.train()

        rescale_xy = torch.ones((1, 4, 1, 1)).to(args.device)
        rescale_xy[:, 0] = args.max_x_velocity
        rescale_xy[:, 1] = args.max_y_velocity
        rescale_xy[:, 2] = args.max_x
        rescale_xy[:, 3] = args.max_y

        class_criterion = CrossEntropyLoss(ignore_index=0) # we let 0 represent the unknown lane id
        observed_last = args.t_h // args.down_sampling_steps
        for iteration, (ori_data, neighbor_matrices, _, _) in enumerate(train_loader):
            # ori_data: (N, C, T, V), target_vehicle_ids: (N,)
            # C = 7 : [dataset_id, vehicle_id, frame_id local_x, local_y, lane_id] + [mark]
            # for now_history_frames in range(5, args.t_h // args.down_sampling_steps + 2):
            # now_history_frames = args.t_h // args.down_sampling_steps + 1 # 30 // 2 + 1, 30: history frames, 2: down sampling steps
            data, no_norm_loc_data, neighbor_matrices = preprocess_data(ori_data, rescale_xy, neighbor_matrices, observed_last=observed_last)
            # data, _, _, A = preprocess_data(ori_data, rescale_xy, neighbor_matrices)
            #TODO change the range, decrease the training overload
            for now_history_frames in [16]: 
            
                input_data = data[:, :, :now_history_frames, :] # (N, 4, T, V)
                output_loc_GT = data[:, :2, now_history_frames:, :]
                output_mask = no_norm_loc_data[:, -1:, now_history_frames:, :] # (N, 1, T, V)
                input_mask = no_norm_loc_data[:, -1:, :now_history_frames, :]
                A = neighbor_matrices[:, :now_history_frames]

                predicted, _ = model(pra_x=input_data, pra_A=A, input_mask=input_mask, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=teacher_forcing, pra_teacher_location=output_loc_GT)
                # Compute loss for training
                #TODO We use abs to compute loss to backward updata weights
                overall_sum_time, overall_num, _ = compute_RMSE(predicted, output_loc_GT, output_mask, error_order=1)
                # overall_loss
                overall_num = torch.sum(overall_num, axis=0)
                overall_num[overall_num < 1] = 1
                total_loss = torch.sum(overall_sum_time, axis=0) / overall_num # (T, )
                total_loss = total_loss.sum() / total_loss.size(0) #(1,)
                # total_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num), torch.ones(1,).to(args.device)) #(1,)
                total_loss.backward()

            if iteration % 240 == 0:
                now_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                epoch_log='Epoch:{:>4}/{:>4}'.format(i_epoch, args.num_epochs)
                logger.info('|{:>20}|\tIteration:{:>5}|\tLoss:{:.6f}|\tlr: {:.5f}|\ttf: {:.3f}|'.format(epoch_log, iteration, total_loss.data.item(), now_lr, teacher_forcing))            
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
            if iteration % num_iter_to_val == 0 and iteration > 0:

                logger.info('################# Val ##########################')
                # s_t = time.time()
                all_overall_sum, all_overall_num, compare_overall_sum, compare_overall_num = val_model(model, val_loader, num_batch_val)
                all_rmse = display_RMSE_test(
                    (all_overall_sum, all_overall_num),
                    '{}_Epoch{}_Iteration{}'.format('Val_all', i_epoch, iteration)
                )
                compare_rmse = display_RMSE_test(
                    (compare_overall_sum, compare_overall_num),
                    '{}_Epoch{}_Iteration{}'.format('Val_compare', i_epoch, iteration)
                )
                
                # e_t = time.time()
                # logger.info(f'one training epoch and a validation process need {(e_t - s_t) / 60} minutes.')
                rmse = np.sum(compare_rmse)
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


def run_test(model, test_data_path, model_path=None, congest_scene_only=False):
    if model_path:
        model, _ = load_ckpt(model_path, model)
    model.to(args.device)
    logger.info(f'################# Testing ##########################')
    test_loader, num_batch_test = data_loader(
        test_data_path,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        train_val_test='test'
    )
    all_overall_sum, all_overall_num, compare_overall_sum, compare_overall_num = val_model(model, test_loader, num_batch_test, congest_scene_only=congest_scene_only)
    all_rmse = display_RMSE_test(
        (all_overall_sum, all_overall_num),
        '{}'.format('Test_all')
    )
    compare_rmse = display_RMSE_test(
        (compare_overall_sum, compare_overall_num),
        '{}'.format('Test_compare')
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
    
    train_path = 'data/TrainSet.mat'
    dev_path = 'data/ValSet.mat'
    test_path = 'data/TestSet.mat'

    args.model_name = 'STransformerRNN'
    args.predict_velocity = True

    # Set logger
    args.identity = f'{args.model_name}_{int(args.spatial_interact)}_{int(args.transformer_layers)}_{int(args.interact_in_decoding)}_{args.seq2seq_type}_train_plus_val' + \
                    f'_{args.base_learning_rate}_{args.data_aug_ratio}_{args.teacher_forcing}'
    log_file = os.path.join(log_dir,  f'{args.identity}_.log')
    logger.add(log_file, format="<green>{time}</green> <level>{message}</level>", level="INFO")

    model = SpatialTransformerRNN(
        in_size=4, out_size=2, seq2seq_type='gru', 
        n_layers=args.transformer_layers, 
        interact_in_decoding=args.interact_in_decoding,
        spatial_interact=args.spatial_interact
    )
    args.max_x_velocity = 14.85
    args.max_x =  36.155
    args.max_y_velocity = 67.58
    args.max_y = 486.76
    resume_training = False
    # args.resume_train_model_path = 'STransformerRNN_1_4_0_gru_train_plus_val_0.0001_0.0_0.8_0_74036.pt'
    if len(args.resume_train_model_path) > 0:
        resume_training = True
    args.resume_train_model_path = f'outputs/saved_models/{args.resume_train_model_path}'
    logger.info(f'Global arguments: {args.__dict__}')
    if args.do_train:
        run_train_val(model, train_data_path=train_path, dev_data_path=dev_path, test_data_path=test_path, resume_training=resume_training)
    if args.do_test:
        model_path = 'outputs/saved_models/STransformerRNN_1_4_0_gru_train_plus_val_0.0001_0.0_0.8_2_37018.pt'
        run_test(model, test_data_path=test_path, model_path=model_path, congest_scene_only=True)
        



