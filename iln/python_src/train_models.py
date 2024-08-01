import argparse
import yaml
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from datetime import datetime
import time
import glob
import numpy as np
import sys
from infer import rimg_inference
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Datasets
from dataset.range_images_dataset import RangeImagesDataset
from dataset.samples_from_image_dataset import SamplesFromImageDataset
from dataset.dataset_utils import generate_dataset

# Models
from models.iln.iln import ILN
from models.liif_cvpr21.liif_lidar import LIIFLiDAR
from models.lsr_ras20.unet import UNet
from models.model_utils import generate_model

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import sys

def is_valid_check_point():
    if check_point['model']['name'] != config['model']['name']:
        return False

    for key, value in check_point['model']['args'].items():
        print('key, value:', key, value)
        if value != config['model']['args'][key]:
            return False

    if check_point['lidar_in'] != train_dataset.lidar_in:
        return False

    return True

def print_log(epoch, loss_sum, loss_avg,  val_loss_sum = 999.0, val_loss_avg = 999.0,directory=None):
    log_msg = ('%03d %.4f %.4f %.4f %.4f' % (epoch, loss_sum, loss_avg, val_loss_sum, val_loss_avg))
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'training_loss_history.txt'), 'a') as f:
        f.write(log_msg + '\n')

    return print(log_msg)


def save_check_point(best_loss, epoch, loss = 0.0000, period=10):
    if epoch % period == period-1 or best_loss==True:
        check_point_model_info = {'name': config['model']['name'],
                                  'args': config['model']['args'],
                                  'state_dict': model.state_dict() if n_gpus <= 1 else model.module.state_dict()}
        check_point = {'epoch': epoch + 1,
                       'model': check_point_model_info,
                       'optimizer': optimizer.state_dict(),
                       'lr_scheduler': lr_scheduler.state_dict(),
                       'lidar_in': train_dataset.lidar_in}
    #print('Loss!!:',loss)
    if best_loss==True :
        # print('best Loss!!:',loss , '| before was :',best_loss)
        
        
        [os.remove(f) for f in glob.glob(model_directory+'/*best.pth')]
         
        check_point_filename = os.path.join(model_directory, model_name + '_' + str(epoch + 1) + '_best.pth')
        torch.save(check_point, check_point_filename)
    elif epoch % period == period-1:
        # print('Not advanced!!:',loss)
        check_point_filename = os.path.join(model_directory, model_name + '_' + str(epoch + 1) + '.pth')
        torch.save(check_point, check_point_filename)
    return


def train_implicit_network():
    best_loss=9999.0
    
    for epoch in range(epoch_start, epoch_end, 1):     
        f = open('./log.txt', 'w')   
        best_loss_tf=False
        loss_sum = 0.0
        valid_loss_sum = 0.0
        model.train()

        for input_range_images, input_queries, output_ranges in tqdm(train_loader, file = sys.stdout, desc='train'):
            # input_range_images == [28, 1, 32, 1024]
            # input_queries == [28, 32768, 2]

            # Load data: [-1 ~ 1]
            input_range_images = input_range_images.cuda()
            # print(input_range_images.shape)
            # print("\n train data = {}".format(input_range_images))
            input_queries = input_queries.cuda()

            # Initialize gradient
            optimizer.zero_grad()

            # Prediction
            pred_ranges = model(input_range_images, input_queries) # pred_ranges == [28, 32768, 1]
            

            # Loss
            output_ranges = output_ranges.cuda()
            loss = criterion(pred_ranges, output_ranges)
            f.write('Epoch: '+str(epoch)+ ' Train Loss:'+str(round(float(loss.detach().cpu()),4))+'\n')
            #print('Train Loss:',round(float(loss.detach().cpu()),4), file=sys.stdout)

            # Back-propagation
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.detach().cpu()
            
        model.eval()
        for input_range_images, input_queries, output_ranges in tqdm(valid_loader, leave=False, file = sys.stdout, desc='valid'):
            # Load data: [-1 ~ 1]
            input_range_images = input_range_images.cuda()
            input_queries = input_queries.cuda()
            # Prediction
            pred_ranges = model(input_range_images, input_queries)
            # Loss
            output_ranges = output_ranges.cuda()
            valid_loss = criterion(pred_ranges, output_ranges)
            f.write('Epoch: '+str(epoch)+' Vaild Loss: '+str(round(float(valid_loss.detach().cpu()),4))+'\n')
            #print('Vaild Loss:',round(float(valid_loss.detach().cpu()),4), file=sys.stdout)

            valid_loss_sum += valid_loss.detach().cpu()
        
        # Schedule the learning rate
        lr_scheduler.step()
        print(lr_scheduler.state_dict())

        # # Logging
        print_log(epoch, loss_sum, loss_sum / len(train_loader), valid_loss_sum, valid_loss_sum / len(valid_loader),directory=model_directory)
        if valid_loss_sum < best_loss:
            best_loss=valid_loss_sum
            best_loss_tf = True
        else:
            best_loss_tf = False
        save_check_point(best_loss_tf, epoch, valid_loss_sum / len(valid_loader), period=100)
        f.close()
    return


if __name__ == '__main__':
    # sys.stdout = open('test.txt', 'a')
    print('/opt/conda/bin/python',' '.join(sys.argv))
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Train a LiDAR super-resolution network")
    parser.add_argument('-c', '--config',
                        type=str,
                        required=True,
                        help='Configuration filename. [.yaml]')
    parser.add_argument('-b', '--batch',
                        type=int,
                        required=False,
                        default=64,
                        help='Batch size for network training. (default: 16)')
    parser.add_argument('-cp', '--checkpoint',
                        type=str,
                        required=False,
                        default=None,
                        help='Check point filename. [.pth]')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    
    args = parser.parse_args()

    # Load the configurations
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    n_gpus = torch.cuda.device_count()
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # Train settings
    batch_size = args.batch
    train_dataset = generate_dataset(config['dataset'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    
    # Valid settings 
    valid_dataset = generate_dataset(config['dataset_val'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    model = generate_model(config['model']['name'], config['model']['args'])
    optimizer = optim.Adam(params=list(model.parameters()), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[200, 400, 600 , 800], gamma=0.5)
    criterion = nn.L1Loss()
    epoch_start = 0
    epoch_end = 400

    # load pth 
    # Load a valid check point
    check_point = torch.load(args.checkpoint) if args.checkpoint is not None else None
    if check_point is not None:
        if is_valid_check_point():
            model.load_state_dict(check_point['model']['state_dict'])
            epoch_start = check_point['epoch']
            optimizer.load_state_dict(check_point['optimizer'])
            lr_scheduler.load_state_dict(check_point['lr_scheduler'])
        else:
            print('ERROR: Invalid check point file:', args.checkpoint)
            exit(0)

    # Set the multi-gpu for training
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)
        #dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        #model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    model.cuda()

    print("=================== Training Configuration ====================  ")
    model_name = config['model']['name']
    model_directory = config['model']['output']
    print('  Model:', model_name, '(' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + ' parameters)')
    for key, value in config['model']['args'].items():
        print('    ' + key + ':', value) #key == d, h  
    print('  Output directory:', model_directory)
    print('  Check point file:', args.checkpoint)
    print('  ')
    print('  Dataset:', config['dataset']['name'], '[' + config['dataset']['type'] + '] (' + str(len(train_dataset)) + ' pairs)')
    print('  Batch:', batch_size)
    print('  Epoch:', epoch_start, '-->', epoch_end)
    print('  GPUs:', n_gpus)
    print("===============================================================  \n")
    start_time = str(datetime.now())
    print(start_time)
    train_implicit_network()
    end_time = str(datetime.now())
    print(end_time)


