

# -*- coding: utf-8 -*-
import argparse
import os
import copy
import struct
import shutil

import torch
import torch.nn as nn
import numpy as np
import glob
from tqdm import tqdm
import itertools

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
V_channel = 32
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
        if ch+1 == 32:
            break
        range_image[ch][j] = range_image[ch+1][j]
    
    return range_image

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

    range_image = range_image.transpose()
    range_image = range_image.astype(np.float32)
    # range_image = np.flip(range_image, axis=0)
    # range_image = np.flip(range_image, axis=1)
    range_image_file.close()
    range_image[range_image==100] = 0
    range_image = denoise(range_image)
    # print(range_image.shape)
    # range_image = outlier_delete(denoise(range_image))

    return range_image


def get_pcd() : 
    test_dir_num = ['004', '005', '018', '032', '048', '059', '061', '065', '072', '073', '080', '114', '115']
    
    file_list_test_128 = list()
    file_list_test_32 = list()
    
    for i in test_dir_num : 
        file_list_test_128 = file_list_test_128 + sorted(glob.glob('../../super_resolution_data/49-1_Super_Resolution/High_Resolution/img/sequences/'+i+'/velodyne/*.pcd'))
        file_list_test_32 = file_list_test_32 + sorted(glob.glob('../../super_resolution_data/49-1_Super_Resolution/Low_Resolution/img/sequences/'+i+'/velodyne/*.pcd'))
    
    print(len(file_list_test_128))
    output_dir = './inference/pcd/origin_32/'
    
    for i in range(0, len(file_list_test_32)):
        rename = output_dir+str(i)+'.pcd'
        shutil.copyfile(file_list_test_32[i], rename)
        
def mult(image, number):
    start = int(image.shape[1]/24*number)
    end = int(image.shape[1]/24*(number+1))
    # print(number, start, end)
    points = np.empty([0,3])
    for i in range(start,end):

        horizon_index = i - image.shape[1]/2
        # if horizon_index <= 0:
        #     horizon_index = horizon_index*-1
        # else :
        #     horizon_index = horizon_index + image.shape[1]/2
        horizon_angle_rad = 360*horizon_index/image.shape[1] *np.pi/180
        
        for j in range(0,image.shape[0]): # 128 V_channel
            L = image[j,i]
            # if L > 80:
            #     continue
            V_angle_rad = (lidar_vertical_ange_offset*j-lidar_vertical_fov/2) *np.pi/180 * (-1)

            xy_D = L*np.cos(V_angle_rad)
            z = L*np.sin(V_angle_rad)
            y = xy_D*np.sin(horizon_angle_rad)
            x = xy_D*np.cos(horizon_angle_rad)

            points = np.append(points, np.array([[x,y,z]]),axis=0)
    return points
    
def rimg2pcd(input_rimg_dir, output_pcd_dir):
    if not os.path.exists(output_pcd_dir):
                os.makedirs(output_pcd_dir)
    
    input_rimg_file = sorted(glob.glob(input_rimg_dir+'/*.rimg'))
    print(len(input_rimg_file))

    print('start transform pcd!!')

    for i in tqdm(range(0, len(input_rimg_file))) : 
        # input_rimg_file[i] : 대상 Range_iamge 경로
        output_pcd_file = output_pcd_dir + '/' + str(i) +'.pcd'

        image = read_range_image(input_rimg_file[i])
        file_name = input_rimg_file[i].split('/')[-1] #
        # plt.imshow(X=image)
        # plt.show()
        
        ## 멀티 쓰레딩 Pool 사용
        number = range(0,24)
        pool = multiprocessing.Pool(processes=24) # 현재 시스템에서 사용 할 프로세스 개수
        func = partial(mult, image)
        result = pool.map(func, number)

        pool.close()
        pool.join()

        points = np.empty([0,3])
        for i in range(0,24):
            points = np.append(points, result[i], axis=0)

        ## pcd 저장
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(output_pcd_file, pcd)

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Demonstrate the LiDAR image super-resolution to a target resolution")
    parser.add_argument('-r', '--target_resolution',
                        type=str,
                        required=False,
                        default='32_1024',
                        help='Vertical and horizontal target resolution; (default: 128_2048)')
    args = parser.parse_args()

    rimg_file_list_downsample = sorted(glob.glob('./rimg_dataset/Custom_2/Lidar_sr_data/32_1024_downsample/test/*.rimg'))

    output_pcd_dir_downsample = './inference/pcd/downsample_32/'
    denoise_rimg_downsample = './rimg_dataset/Custom_2/Lidar_sr_data/32_1024_downsample/test/' 

    check_point_pth = glob.glob('./models/trained/iln_0d/pretrain_weight.pth') 

    rimg2pcd(denoise_rimg_downsample, output_pcd_dir_downsample)
# get_pcd()
