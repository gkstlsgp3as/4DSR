import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import time
from multiprocessing import Pool
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime


def read_range_image_binary(input_image, label_image, dtype=np.float16, lidar=None):
    # Read the range image
    input_image_file = open(input_image, 'rb')
    print('\nInput data:', input_image_file)
    label_image_file = open(label_image, 'rb')
    print('Label data:', label_image_file)
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


if __name__ == '__main__':  
    print('/opt/conda/bin/python',' '.join(sys.argv))
    input_rimg_file_list = sorted(glob.glob('./inference/rimg/denoise/origin_32_to_128/denoise_output_*.rimg'))
    label_image_file_list = sorted(glob.glob('./rimg_dataset/Custom_2/Lidar_sr_data/128_1024_origin/test/*.rimg'))
    start_time = str(datetime.now())
    print(start_time)

    # start = time.time()
    iou_total = 0
    MAE_total = 0
    RMSE_total = 0
    iou_list = list()
    MAE_list = list()
    RMSE_list = list()
    for input_file, label_file in tqdm(zip(input_rimg_file_list, label_image_file_list), file = sys.stdout, desc='eval') :
        iou_, MAE_, RMSE_= read_range_image_binary(input_file, label_file)
        
        iou_ = float(iou_)
        MAE_ = float(MAE_)
        RMSE_ = float(RMSE_)
        
        print('\niou:', iou_)
        print('MAE', MAE_)
        print('RMSE:', RMSE_)

        iou_total += iou_
        MAE_total += MAE_
        RMSE_total += RMSE_
        iou_list.append(iou_)
        MAE_list.append(MAE_)
        RMSE_list.append(RMSE_)
   
    iou_avg = iou_total/len(input_rimg_file_list)
    MAE_avg = MAE_total/len(input_rimg_file_list)
    RMSE_avg = RMSE_total/len(input_rimg_file_list)

    print('iou_avg:', iou_avg)
    print('MAE_avg:', MAE_avg)
    print('RMSE_avg:', RMSE_avg)
    
    end_time = str(datetime.now())
    print(end_time)



