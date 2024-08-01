import argparse
import yaml
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from infer import rimg_inference
# from evaluate import print_eval
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

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

rimg_file_list_origin = sorted(glob.glob('./rimg_dataset/Custom_2/Lidar_sr_data/32_1024_origin/test/*.rimg'))
rimg_file_list_downsample = sorted(glob.glob('./rimg_dataset/Custom_2/Lidar_sr_data/32_1024_downsample/test/*.rimg'))

output_directory_origin = './inference/rimg/origin_32_to_128/'
output_directory_downsample = './inference/rimg/downsample_32_to_128/'

output_pcd_dir_origin = './inference/pcd/origin_32_to_128/'
output_pcd_dir_downsample = './inference/pcd/downsample_32_to_128/'

denoise_rimg_origin = './inference/rimg/denoise/origin_32_to_128/'
denoise_rimg_downsample = './inference/rimg/denoise/downsample_32_to_128/'
# denoise_rimg_origin = './inference/rimg/origin_32_to_128/'
# denoise_rimg_downsample = './inference/rimg/downsample_32_to_128/'

check_point_pth = glob.glob('./models/trained/iln_0d/*best.pth')

# input_rimg_file_list = sorted(glob.glob('./inference/rimg/denoise/origin_32_to_128/denoise_output_*.rimg'))
# label_image_file_list = sorted(glob.glob('./rimg_dataset/Custom_2/Lidar_sr_data/128_1024_origin/test/*.rimg'))

input_rimg_file_list = sorted(glob.glob('./inference/rimg/origin_32_to_128/output_*.rimg'))
label_image_file_list = sorted(glob.glob('./rimg_dataset/Custom_2/Lidar_sr_data/128_1024_origin/test/*.rimg'))


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

def read_range_image_binary_eval(input_image, label_image, dtype=np.float16, lidar=None):
    # Read the range image
    input_image_file = open(input_image, 'rb')
    label_image_file = open(label_image, 'rb')
    size_1 = np.fromfile(input_image_file, dtype=np.uint64, count=2) 
    input_image = np.fromfile(input_image_file, dtype=dtype)
    input_image = input_image.reshape(size_1[1], size_1[0])

    size_2 = np.fromfile(label_image_file, dtype=np.uint64, count=2)
    label_image = np.fromfile(label_image_file, dtype=dtype)
    label_image = label_image.reshape(size_2[1], size_2[0])
    
#     input_image = input_image.transpose()
    input_image = input_image.astype(np.float32)
    input_image = np.flip(input_image, axis=0)
    input_image_file.close()

    label_image = label_image.transpose()
    label_image = label_image.astype(np.float32)
    label_image = np.flip(label_image, axis=0)
    label_image_file.close()

    
    input_image[input_image==100] = 0
    
    mse = mean_squared_error(label_image, input_image)
    mae = mean_absolute_error(label_image, input_image)
    rmse = mse ** 0.5
    
    IOU_count = 0
    for h in range(0, size_1[1]) : #channel 
        for w in range(0, size_1[0]) :
            if input_image[h][w] == label_image[h][w] :
                IOU_count = IOU_count+1

    iou = "{:.6f}".format(IOU_count/131072)

    
    return iou, mae, rmse

def print_eval(input_rimg_file_list, label_image_file_list, epoch, period=100):
    if epoch % period == period-1:

        iou_total = 0
        MAE_total = 0
        RMSE_total = 0
        iou_list = list()
        MAE_list = list()
        RMSE_list = list()
        for input_file, label_file in tqdm(zip(input_rimg_file_list, label_image_file_list)) :
            iou_, MAE_, RMSE_= read_range_image_binary_eval(input_file, label_file)
            
            iou_ = float(iou_)
            MAE_ = float(MAE_)
            RMSE_ = float(RMSE_)

            iou_total += iou_
            MAE_total += MAE_
            RMSE_total += RMSE_
            iou_list.append(iou_)
            MAE_list.append(MAE_)
            RMSE_list.append(RMSE_)
    
        iou_avg = iou_total/len(input_rimg_file_list)
        MAE_avg = MAE_total/len(input_rimg_file_list)
        RMSE_avg = RMSE_total/len(input_rimg_file_list)

        print('iou:', iou_avg)
        print('MAE:', MAE_avg)
        print('RMSE:', RMSE_avg)


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
        print('best Loss!!:',loss , '| before was :',best_loss)
        
        
        [os.remove(f) for f in glob.glob(model_directory+'/*best.pth')]
         
        check_point_filename = os.path.join(model_directory, model_name + '_' + str(epoch + 1) + '_best.pth')
        torch.save(check_point, check_point_filename)
    elif epoch % period == period-1:
        print('Not advanced!!:',loss)
        check_point_filename = os.path.join(model_directory, model_name + '_' + str(epoch + 1) + '.pth')
        torch.save(check_point, check_point_filename)
    return

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
        print('best Loss!!:',loss , '| before was :',best_loss)
        
        
        [os.remove(f) for f in glob.glob(model_directory+'/*best.pth')]
         
        check_point_filename = os.path.join(model_directory, model_name + '_' + str(epoch + 1) + '_best.pth')
        torch.save(check_point, check_point_filename)
    elif epoch % period == period-1:
        print('Not advanced!!:',loss)
        check_point_filename = os.path.join(model_directory, model_name + '_' + str(epoch + 1) + '.pth')
        torch.save(check_point, check_point_filename)
    return

def train_implicit_network():
    best_loss=9999.0
    for epoch in range(epoch_start, epoch_end, 1):
        best_loss_tf=False
        loss_sum = 0.0
        valid_loss_sum = 0.0
        model.train()
        for input_range_images, input_queries, output_ranges in tqdm(train_loader, leave=False, desc='train'):
            # Load data: [-1 ~ 1]
            input_range_images = input_range_images.cuda()
            
            input_queries = input_queries.cuda()

            # Initialize gradient
            optimizer.zero_grad()

            # Prediction
            pred_ranges = model(input_range_images, input_queries)

            # Loss
            output_ranges = output_ranges.cuda()
            loss = criterion(pred_ranges, output_ranges)

            # Back-propagation
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.detach().cpu()
        model.eval()
        for input_range_images, input_queries, output_ranges in tqdm(valid_loader, leave=False, desc='valid'):
            # Load data: [-1 ~ 1]
            input_range_images = input_range_images.cuda()
            input_queries = input_queries.cuda()

            # Prediction
            pred_ranges = model(input_range_images, input_queries)

            # Loss
            output_ranges = output_ranges.cuda()
            valid_loss = criterion(pred_ranges, output_ranges)

            valid_loss_sum += valid_loss.detach().cpu()
        
        # Schedule the learning rate
        lr_scheduler.step()

        # Logging
        print_log(epoch, loss_sum, loss_sum / len(train_loader), valid_loss_sum, valid_loss_sum / len(valid_loader),directory=model_directory)
        if valid_loss_sum < best_loss:
            best_loss=valid_loss_sum
            best_loss_tf = True
        else:
            best_loss_tf = False
        save_check_point(best_loss_tf, epoch, valid_loss_sum / len(valid_loader), period=100)
        rimg_inference(rimg_file_list_origin, output_directory_origin, check_point_pth[0], epoch, period=100)
        print_eval(input_rimg_file_list, label_image_file_list, epoch, period=100)

    return


def train_pixel_based_network():
    for epoch in range(epoch_start, epoch_end, 1):
        loss_sum = 0.0
        valid_loss = 0.0
        model.train()
        for low_res, high_res in tqdm(train_loader, leave=False, desc='train'):
            # Load data: [-1 ~ 1]
            low_res = low_res.cuda()
            high_res = high_res.cuda()

            # Initialize gradient
            optimizer.zero_grad()

            # Prediction
            low_res = (low_res + 1.0) * 0.5                 # [-1 ~ 1] -> [ 0 ~ 1]
            pred_high_res = model(low_res)
            pred_high_res = (pred_high_res * 2.0) - 1.0     # [ 0 ~ 1] -> [-1 ~ 1]

            # Loss
            loss = criterion(pred_high_res, high_res)

            # Back-propagation
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.detach().cpu()
            
        #########################################validation
        model.eval()
        for low_res, high_res in tqdm(valid_loader, leave=False, desc='valid'):
            # Load data: [-1 ~ 1]
            low_res = low_res.cuda()
            high_res = high_res.cuda()

            # Initialize gradient
            optimizer.zero_grad()

            # Prediction
            low_res = (low_res + 1.0) * 0.5                 # [-1 ~ 1] -> [ 0 ~ 1]
            pred_high_res = model(low_res)
            pred_high_res = (pred_high_res * 2.0) - 1.0     # [ 0 ~ 1] -> [-1 ~ 1]

            # Loss
            valid_loss = criterion(pred_high_res, high_res)

            # Back-propagation
            valid_loss_sum = valid_loss_sum + loss.detach().cpu()

        # Logging
        print_log(epoch, loss_sum, loss_sum / len(train_loader),  valid_loss_sum, valid_loss_sum / len(train_loader), directory=model_directory)
        save_check_point(epoch, valid_loss_sum / len(valid_loader), period=10 )


    return


if __name__ == '__main__':
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
                        # default='./models/trained/iln_0d/pretrain_weight.pth',
                        help='Check point filename. [.pth]')
    args = parser.parse_args()

    # Load the configurations
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    n_gpus = torch.cuda.device_count()

    # Train settings
    batch_size = args.batch
    train_dataset = generate_dataset(config['dataset'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    # Valid settings 
    valid_dataset = generate_dataset(config['dataset_val'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    
    model = generate_model(config['model']['name'], config['model']['args'])#.train()
    optimizer = optim.Adam(params=list(model.parameters()), lr=2e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[200, 400, 600 , 800], gamma=0.5)
    criterion = nn.L1Loss()
    epoch_start = 0
    epoch_end = 1000


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

    # NOTE: The training approaches are different according to the type of network structure
    if config['dataset']['type'] == 'range_images':
        train_pixel_based_network()
    elif config['dataset']['type'] == 'range_samples_from_image': #now, here
        train_implicit_network()


