#
# Module:       pcd2bin.py
# Description:  .pcd to .bin converter
#
# Author:       Yuseung Na (ys.na0220@gmail.com)
# Version:      1.0
#
# Revision History
#       January 19, 2021: Yuseung Na, Created
#

import numpy as np
import os
import argparse
# import pypcd
from pypcd import pypcd
import csv
from tqdm import tqdm
from glob import glob
import shutil

def main():
    ## Add parser
    parser = argparse.ArgumentParser(description="Convert .pcd to .bin")
    parser.add_argument(
        "--pcd_path",
        help=".pcd file path.",
        
        type=str,
        default="/home/user/lidar_pcd"
    )
    parser.add_argument(
        "--bin_path",
        help=".bin file path.",
        type=str,
        default="/home/user/lidar_bin"
    )
    parser.add_argument(
        "--file_name",
        help="File name.",
        type=str,
        default="file_name"
    )
    args = parser.parse_args()

    ## Find all pcd files
    pcd_files = []
    for (path, dir, files) in os.walk(args.pcd_path):
        for filename in files:
            # print(filename)
            ext = os.path.splitext(filename)[-1]
            if ext == '.pcd':
                pcd_files.append(path + "/" + filename)

    ## Sort pcd files by file name
    pcd_files.sort()   
    print("Finish to load point clouds!")

    ## Make bin_path directory
    try:
        if not (os.path.isdir(args.bin_path)):
            os.makedirs(os.path.join(args.bin_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print ("Failed to create directory!!!!!")
            raise

    ## Generate csv meta file
    csv_file_path = os.path.join(args.bin_path, "meta.csv")
    csv_file = open(csv_file_path, "w")
    meta_file = csv.writer(
        csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )
    ## Write csv meta file header
    meta_file.writerow(
        [
            "pcd file name",
            "bin file name",
        ]
    )
    print("Finish to generate csv meta file")

    root_dir = args.pcd_path 
    pcd_dir = root_dir+'img/sequences'
    label_dir = root_dir+'label/sequences'
    
    save_path = args.bin_path
    seq_dirs = glob(pcd_dir + '/*')
    seq_dirs.sort()
    print(pcd_dir)
    for seq_dir in seq_dirs:
        seq_pcd_paths = glob(seq_dir + '/velodyne/*')
        #print(seq_dir + '/velodyne/*')
        seq_pcd_paths.sort()
        seq_dir_name = seq_dir.split('/')[-1]
        seq = 0
        if not os.path.exists(save_path + '/sequences/' + seq_dir_name + '/velodyne'):
            os.makedirs(save_path + '/sequences/' + seq_dir_name + '/velodyne')
        if not os.path.exists(save_path + '/sequences/'+ seq_dir_name + '/labels'):
            os.makedirs(save_path + '/sequences/'+ seq_dir_name + '/labels')        
        for seq_pcd_path in tqdm(seq_pcd_paths):
            #try:
            #print(seq_pcd_path)
            real_name = seq_pcd_path.split('/')[-1].replace('.pcd','')
            label_file_name = "{:05d}.label".format(seq)
            label_file_path = save_path + '/sequences/'+ seq_dir_name + '/labels/' + label_file_name
            ori_label_path = label_dir + '/' + seq_dir_name + '/labels/' + real_name +'.label'
            #print(ori_label_path,':',label_file_path)
            shutil.copyfile(ori_label_path, label_file_path) 
            pc = pypcd.PointCloud.from_path(seq_pcd_path)
            bin_file_name = "{:05d}.bin".format(seq)
            bin_file_path = os.path.join(args.bin_path, bin_file_name)
            np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
            np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
            np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
            np_i = (np.array(pc.pc_data['reflectivity'], dtype=np.float32)).astype(np.float32)/256

            points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
            points_32.tofile(save_path + '/sequences/' + seq_dir_name + '/velodyne/' + bin_file_name)
            seq = seq + 1
            #except:
            #    with open('missing_data.txt','a') as f:
            #        f.write(real_name+'\n')
            
    
if __name__ == "__main__":
    main()
