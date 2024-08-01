import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import struct
import os

image_cols = 2048
ang_res_x = 360.0/float(image_cols)#360.0/float(image_cols) # horizontal resolution
ang_y = 45 #22.5
max_range = 100 #80.0
min_range = 5 #2.0
ang_start_y = ang_y/2 # bottom beam angle

## 여기도 원하시면 바꾸세요!
train_output_path_origin_32 = './rimg_dataset/sample/Lidar_sr_data/32_2048_origin/train/'

def make_range_image(file_list, file_resolution, data_type):
    ang_res_y = ang_y/float(file_resolution-1) # vertical resolution

    for i in tqdm(range(0, len(file_list))) :
        range_image = np.zeros((file_resolution, image_cols), dtype=np.float32)
        pcd = o3d.io.read_point_cloud(file_list[i])
        points_array = np.asarray(pcd.points).astype(np.float32)
        x = points_array[:,0]
        y = points_array[:,1]
        z = points_array[:,2]
        # find row id
        vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi # radian to angle
        relative_vertical_angle = vertical_angle + ang_start_y
        rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
        # find column id
        horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi # radian to angle
        colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2
        shift_ids = np.where(colId>=image_cols)
        colId[shift_ids] = colId[shift_ids] - image_cols
        # filter range
        thisRange = np.sqrt(x * x + y * y + z * z)
        thisRange[thisRange > max_range] = 0
        thisRange[thisRange < min_range] = 0
        # save range info to range image
        for j in range(len(thisRange)):
            if rowId[j] < 0 or rowId[j] >= file_resolution or colId[j] < 0 or colId[j] >= image_cols:
                continue
            range_image[int(rowId[j]), int(colId[j])] = thisRange[j]

        range_image = range_image.transpose()
        range_image = np.flip(range_image, axis=0) 
        # print(range_image.shape)
        if range_image.shape[1] == 128 :
            if data_type == 'train': 
                output_path = train_output_path_128 
                output_path_downsaple_32 = train_output_path_downsaple_32
            elif data_type == 'valid':
                output_path = valid_output_path_128 
                output_path_downsaple_32 = valid_output_path_downsaple_32
            else :
                output_path = test_output_path_128 
                output_path_downsaple_32 = test_output_path_downsaple_32
        else :
            if data_type == 'train': 
                output_path = train_output_path_origin_32
            elif data_type == 'valid':
                output_path = valid_output_path_origin_32
            else :
                output_path = test_output_path_origin_32
        
        if not os.path.exists(output_path):
                os.makedirs(output_path)
        

        fw = open(output_path + str(i).zfill(5) + '.rimg', 'wb')
        # fw.write(b'\x10\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00')
        size = struct.pack('QQ', range_image.shape[1], range_image.shape[0])
        fw.write(size)
        for width in range(0,range_image.shape[0]):      #1024
            for ch in range(0,range_image.shape[1]):  #32
                L = range_image[width,ch]
                s = np.float16(L).tobytes()
                fw.write(bytes([s[0],s[1]]))
        fw.close()

def read_pcd():
    
    file_list_train_32 = list()

    ## 여기 경로 바꾸세요! 
    file_list_train_32 = sorted(glob.glob('../../super_resolution_data/sample/*.pcd'))#('../../super_resolution_data/Training/01.원천데이터/TS_수집차량_1_lidar_H/*.pcd'))
    print(file_list_train_32)
    
    make_range_image(file_list_train_32, 32, 'train')
    

if __name__ == '__main__':
    read_pcd()







