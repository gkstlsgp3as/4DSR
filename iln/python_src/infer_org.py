# -*- coding: utf-8 -*-
import argparse
import os
import copy

import torch
import torch.nn as nn
import numpy as np
import glob
from tqdm import tqdm
import itertools
import sys
from datetime import datetime
import time

import open3d as o3d
import matplotlib.pyplot as plt

import multiprocessing
from functools import partial

# Datasets
from dataset.dataset_utils import read_range_image_binary, write_range_image_binary, generate_laser_directions
from dataset.dataset_utils import normalization_ranges, denormalization_ranges, normalization_queries

# Models
from models.iln.iln import ILN
from models.liif_cvpr21.liif_lidar import LIIFLiDAR
from models.interpolation.interpolation import Interpolation
from models.model_utils import generate_model

## 수직 channel 번호 설정
V_channel = 128
## 수평 channel 번호 설정
H_channel = 1024

# os2 사양 빔폭 22.5도 
lidar_vertical_fov = 22.5
lidar_vertical_ange_offset = lidar_vertical_fov/(V_channel-1)


def predict_detection_distances(model, input_image, pixel_centers, pred_batch=100000):
    """
    Predict a high-resolution range image (prediction) from low-resolution image (input) and pixel centers (queries).

    :param input_image: low-resolution LiDAR range image
    :param pixel_centers: query lasers associated with pixel centers of range image
    :param pred_batch: batch size for predicting the detection distances (default: 100000)
    :return high-resolution LiDAR range image
    """
    input_image = torch.from_numpy(input_image.copy())[None, None, :, :].cuda()
    pixel_centers = torch.from_numpy(pixel_centers)[None, :, :].cuda()

    with torch.no_grad():
        model.gen_feat(input_image)
        n = pixel_centers.shape[1] #128x2048
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + pred_batch, n) #min(ql + 100000, 128x2048)
            pred = model.query_detection(pixel_centers[:, ql:qr, :])
            preds.append(pred)
            ql = qr

    return torch.cat(preds, dim=1).view(-1).cpu().numpy()

def denoise(range_image): #아랫값 치환 
    ## range_image 입력 --> 노이즈 제거된 range_image 출력
    ## 0인 값 주변의 값으로 변환
    index = np.where(range_image == 0)
    for ch, j in zip(index[0], index[1]) :
        if ch+1 == 128:
            break
        range_image[ch][j] = range_image[ch+1][j]
    
    return range_image

def rimg_inference(rimg_file_list, output_directory, check_point_pth) : #inference 
    if not os.path.exists(output_directory):
                os.makedirs(output_directory)

    for rimg_file in tqdm(rimg_file_list, file = sys.stdout) :
            print('\n[Input range image]:',rimg_file)
            file_name = rimg_file.split('/')[-1]
            # Load the check point
            check_point = torch.load(check_point_pth) #모델 불러오기 
            # Model
            model = generate_model(check_point['model']['name'], check_point['model']['args'])
            model.load_state_dict(check_point['model']['state_dict'], strict=False)
            model.eval().cuda()

            # Prepare the trained LiDAR specification
            lidar_in = check_point['lidar_in']
            lidar_out = copy.deepcopy(check_point['lidar_in'])
            lidar_out['channels'] = 128
            lidar_out['points_per_ring'] = 1024

            check_point_filename = check_point_pth
            model_name = check_point['model']['name']
            input_filename = rimg_file
            
            # Read the input range image (normalized)
            input_range_image = read_range_image_binary(input_filename, lidar=lidar_in)
            input_range_image = normalization_ranges(input_range_image, norm_r=lidar_in['norm_r'])

            # Generate the query lasers (normalized)
            query_lasers = generate_laser_directions(lidar_out)
            query_lasers = normalization_queries(query_lasers, lidar_in)

            # Reconstruct the up-scaled output range image (normalized)
            pred_detection_distances = predict_detection_distances(model,
                                                                input_image=input_range_image,
                                                                pixel_centers=query_lasers,
                                                                pred_batch=100000)
            # pred_detection_distances = predict_detection_distances(input_image=input_range_image, pixel_centers=query_lasers, pred_batch=100000, )
            pred_range_image = pred_detection_distances.reshape(lidar_out['channels'], lidar_out['points_per_ring'])

            # 3. Save the output range image [.rimg]
            output_filename = os.path.join(output_directory, 'output_'+ file_name) if output_directory else None
            print('[Pre_Output]:',output_filename)
            denormalized_pred_range_image = denormalization_ranges(pred_range_image, norm_r=lidar_out['norm_r'])
            denormalized_pred_range_image[denormalized_pred_range_image==0] = 0
            write_range_image_binary(range_image=denormalized_pred_range_image, filename=output_filename)

def outlier_delete(image): #아웃라이어 제거 
    for i in range(0,image.shape[0]):
        # 하나의 채널에서 양 끝 값은 비교하는데 사용하지 않는다
        # Ex) 0~1023 개의 채널 값에서 [0],[1023]번은 사용하지 않음
        # 각각 2개 데이터 지우기
        x_data = np.delete(image[i,:], (0,image.shape[1]-1))       # 기준 데이터 [10]
        front_data = np.delete(image[i,:], (0,1))      # 앞 데이터   [11]
        back_data = np.delete(image[i,:], (image.shape[1]-2,image.shape[1]-1)) # 뒤 데이터   [9]

        front_slope = front_data - x_data # df/dx = [11 - 10]/1
        back_slope = x_data - back_data   # db/dx = [10 - 9]/1
        result_slope = front_slope*back_slope

        index = np.where(result_slope < -2000) #노이즈가 -2000이하에서 생성이됨
        image[i,:][index[0]+1] = 0
    return image

def read_range_image(filename, dtype=np.float16, lidar=None):
    # Read the range image
    range_image_file = open(filename, 'rb') ##
    # 앞에 사이즈 읽기   1. 입력.rimg = np.uint, 2.  출력.rimg = np.uint64
    size = np.fromfile(range_image_file, dtype=np.uint64, count=2) 
    range_image = np.fromfile(range_image_file, dtype=dtype) 
    range_image = range_image.reshape(size[1], size[0]) 

    range_image = range_image.astype(np.float32)
    range_image = np.flip(range_image, axis=0)
    range_image_file.close()
    range_image[range_image==100] = 0
    range_image = denoise(range_image)
    # range_image = outlier_delete(denoise(range_image))

    return range_image


def mult(image, number):
    start = int(image.shape[1]/24*number)
    end = int(image.shape[1]/24*(number+1))
    # print(number, start, end)
    points = np.empty([0,3])
    for i in range(start,end):

        horizon_index = i - image.shape[1]/2
        horizon_angle_rad = 360*horizon_index/image.shape[1] *np.pi/180
        
        for j in range(0,image.shape[0]): # 128 V_channel
            L = image[j,i]
            V_angle_rad = (lidar_vertical_ange_offset*j-lidar_vertical_fov/2) *np.pi/180 * (-1)

            xy_D = L*np.cos(V_angle_rad)
            z = L*np.sin(V_angle_rad)
            y = xy_D*np.sin(horizon_angle_rad)
            x = xy_D*np.cos(horizon_angle_rad)

            points = np.append(points, np.array([[x,y,z]]),axis=0)
    return points

def rimg2pcd(input_rimg_dir, output_pcd_dir, denoise_rimg_path):
    if not os.path.exists(denoise_rimg_path):
                os.makedirs(denoise_rimg_path)
    if not os.path.exists(output_pcd_dir):
                os.makedirs(output_pcd_dir)
    
    input_rimg_file = sorted(glob.glob(input_rimg_dir+'/*.rimg'))

    for i in tqdm(range(0, len(input_rimg_file)),file = sys.stdout) : 
        # input_rimg_file[i] : 대상 Range_iamge 경로
        output_pcd_file = output_pcd_dir + '/' + str(i) +'.pcd'

        image = read_range_image(input_rimg_file[i])
        file_name = input_rimg_file[i].split('/')[-1] #
        output_filename = os.path.join(denoise_rimg_path, 'denoise_'+ file_name) if denoise_rimg_path else None #
        print('[Output range image]:',output_filename)
        
        write_range_image_binary(range_image=image, filename=output_filename) #
        
        ## 멀티 쓰레딩 Pool 사용
        # number = range(0,24)
        # pool = multiprocessing.Pool(processes=24) # 현재 시스템에서 사용 할 프로세스 개수
        # func = partial(mult, image)
        # result = pool.map(func, number)

        # pool.close()
        # pool.join()

        # points = np.empty([0,3])
        # for i in range(0,24):
        #     points = np.append(points, result[i], axis=0)

        # ## pcd 저장
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.io.write_point_cloud(output_pcd_file, pcd)

if __name__ == '__main__':
    print('/opt/conda/bin/python',' '.join(sys.argv))
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Demonstrate the LiDAR image super-resolution to a target resolution")
    parser.add_argument('-r', '--target_resolution',
                        type=str,
                        required=False,
                        default='128_1024',
                        help='Vertical and horizontal target resolution; (default: 128_2048)')
    args = parser.parse_args()

    rimg_file_list_origin = sorted(glob.glob('./rimg_dataset/Custom_2/Lidar_sr_data/32_1024_origin/test/*.rimg'))
    rimg_file_list_downsample = sorted(glob.glob('./rimg_dataset/Custom_2/Lidar_sr_data/32_1024_downsample/test/*.rimg'))

    output_directory_origin = './inference/rimg/origin_32_to_128/'
    output_directory_downsample = './inference/rimg/downsample_32_to_128/'

    output_pcd_dir_origin = './inference/pcd/origin_32_to_128/'
    output_pcd_dir_downsample = './inference/pcd/downsample_32_to_128/'

    denoise_rimg_origin = './inference/rimg/denoise/origin_32_to_128/'
    denoise_rimg_downsample = './inference/rimg/denoise/downsample_32_to_128/'

    # check_point_pth = glob.glob('./models/trained/iln_0d/*best.pth')  
    check_point_pth = glob.glob('./models/trained/iln_0d/pretrain_weight.pth')

    start_time = str(datetime.now())
    print(start_time)
    
    #model inference output 
    rimg_inference(rimg_file_list_origin, output_directory_origin, check_point_pth[0])
    # rimg_inference(rimg_file_list_downsample, output_directory_downsample, check_point_pth[0]) 

    #노이즈 제거 and pcd 생성 
    rimg2pcd(output_directory_origin, output_pcd_dir_origin, denoise_rimg_origin)
    # rimg2pcd(output_directory_downsample, output_pcd_dir_downsample, denoise_rimg_downsample)
    
    end_time = str(datetime.now())
    print(end_time)

  
    

