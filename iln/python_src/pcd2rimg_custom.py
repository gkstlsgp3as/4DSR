import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import struct
import os

image_cols = 1024
ang_res_x = 90.0/float(image_cols)#360.0/float(image_cols) # horizontal resolution
ang_y = 90 #22.5
max_range = 7 #80.0
min_range = 1 #2.0
ang_start_y = ang_y/2 # bottom beam angle

train_output_path_128 = './rimg_dataset/Custom_2/Lidar_sr_data/128_1024_origin/train/'
train_output_path_origin_32 = './rimg_dataset/RETINA/Lidar_sr_data/32_1024_origin/train/'
train_output_path_downsaple_32 = './rimg_dataset/Custom_2/Lidar_sr_data/32_1024_downsample/train/'

valid_output_path_128 = './rimg_dataset/Custom_2/Lidar_sr_data/128_1024_origin/valid/'
valid_output_path_origin_32 = './rimg_dataset/RETINA/Lidar_sr_data/32_1024_origin/valid/'#'./rimg_dataset/Custom_2/Lidar_sr_data/32_1024_origin/valid/'
valid_output_path_downsaple_32 = './rimg_dataset/Custom_2/Lidar_sr_data/32_1024_downsample/valid/'

#test_output_path_128 = './rimg_dataset/Custom_2/Lidar_sr_data/128_1024_origin/test/'
#test_output_path_origin_32 = './rimg_dataset/Custom_2/Lidar_sr_data/32_1024_origin/test/'
#test_output_path_downsaple_32 = './rimg_dataset/Custom_2/Lidar_sr_data/32_1024_downsample/test/'

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
    #train_dir_num = ['000', '001', '002','003','006','007','008','011','012','013','014','015','017','019','020','021','022','023','025','026','027','028','029','030','034','035','036','037','038','039','040','041','042','044','045','046','047','049','050','051','052','053','054','055','056','057','058','060','062','063','064','066','067','068','069','070','071','074','075','076','077','078','079','081','082','083','084','085','086','087','088','089','090','091','092','093','094','095','096','097','100','101','102','103','106','108','109','110','111','112','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131'] #, 
    #valid_dir_num = ['009', '010', '016', '024', '031', '033', '043', '098', '099', '104', '105', '107', '113'] #
    #test_dir_num = ['004', '005', '018', '032', '048', '059', '061', '065', '072', '073', '080', '114', '115']
    

    file_list_train_128 = list()
    file_list_train_32 = list()

    file_list_valid_128 = list()
    file_list_valid_32 = list()

    #file_list_test_128 = list()
    #file_list_test_32 = list()

    #for i in train_dir_num : 
    file_list_train_32 = sorted(glob.glob('../../super_resolution_data/RETINA/*.pcd'))#('../../super_resolution_data/Training/01.원천데이터/TS_수집차량_1_lidar_H/*.pcd'))
    #file_list_train_32 = sorted(glob.glob('../../super_resolution_data/Training/01.원천데이터/TS_수집차량_1_lidar_L/*.pcd'))

    #for i in valid_dir_num : 
    file_list_valid_128 = sorted(glob.glob('../../super_resolution_data/Validation/01.원천데이터/VS_수집차량_1_lidar_H/*.pcd'))
    file_list_valid_32 = sorted(glob.glob('../../super_resolution_data/Validation/01.원천데이터/VS_수집차량_1_lidar_L/*.pcd'))

    #for i in test_dir_num : 
    #    file_list_test_128 = file_list_test_128 + sorted(glob.glob('../../super_resolution_data/49-1_Super_Resolution/High_Resolution/img/sequences/'+i+'/velodyne/*.pcd'))
    #    file_list_test_32 = file_list_test_32 + sorted(glob.glob('../../super_resolution_data/49-1_Super_Resolution/Low_Resolution/img/sequences/'+i+'/velodyne/*.pcd'))

    #make_range_image(file_list_train_128, 128, 'train')
    #make_range_image(file_list_valid_128, 128, 'valid')
    #make_range_image(file_list_test_128, 128, 'test')
    make_range_image(file_list_train_32, 32, 'train')
    #make_range_image(file_list_valid_32, 32, 'valid')
    #make_range_image(file_list_test_32, 32, 'test')


if __name__ == '__main__':
    read_pcd()







