import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import struct
import os
import argparse

image_cols = 1024
ang_res_x = 90.0/float(image_cols)#360.0/float(image_cols) # horizontal resolution
ang_y = 90 #22.5
max_range = 80.0
min_range = 1 #2.0
ang_start_y = ang_y/2 # bottom beam angle



def make_range_image(file_list, file_resolution, output_path):
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

def read_pcd(opt):
    file_list = list()
    file_list = sorted(glob.glob(opt.path+'/*.pcd'))

    make_range_image(file_list, opt.resol, opt.out_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='', help='input image path')
    parser.add_argument('--out-path', type=str, default='', help='output image path')
    parser.add_argument('--resol', type=int, default=32, help='input image resolution')
    opt = parser.parse_args()

    read_pcd(opt)

    # python pcd2rimg_single.py --path "/data/4DRADAR/super_resolution/railSAR_data/data3_Azimuth_느리게" --out-path "/data/4DRADAR/super_resolution/railSAR_data/data3_Azimuth_느리게" --resol 32







