import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# pcd 파일 있는 디렉토리
# pcd_dir_name = "./pcd/pcd_backup/1K_A04_O032_clear_night_summer_03020348.pcd"
pcd_dir_name = "./pcd/pcd_backup/1K_A04_O128_clear_night_summer_03020348.pcd"
pcd = o3d.io.read_point_cloud(pcd_dir_name)
points_array = np.asarray(pcd.points).astype(np.float32)

## 라이다 파라미터
## range image size, depends on your sensor, i.e., VLP-16: 16x1800, OS1-64: 64x1024
# image_rows_full = 32
image_rows_full = 128

image_cols = 1024



## Ouster OS1-64 (gen1)
ang_res_x = 360.0/float(image_cols) # horizontal resolution

ang_y = 22.5
ang_res_y = ang_y/float(image_rows_full-1) # vertical resolution
ang_start_y = ang_y/2 # bottom beam angle

max_range = 80.0
min_range = 2.0


# project points to range image
# range_image = np.zeros((1, image_rows_full, image_cols, 1), dtype=np.float32)
range_image = np.zeros((image_rows_full, image_cols), dtype=np.float32)
x = points_array[:,0]
y = points_array[:,1]
z = points_array[:,2]
# find row id
vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
relative_vertical_angle = vertical_angle + ang_start_y
rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
# find column id
horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2;
shift_ids = np.where(colId>=image_cols)
colId[shift_ids] = colId[shift_ids] - image_cols
# filter range
thisRange = np.sqrt(x * x + y * y + z * z)
thisRange[thisRange > max_range] = 0
thisRange[thisRange < min_range] = 0
# save range info to range image
for i in range(len(thisRange)):
    if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
        continue
    # range_image[0, int(rowId[i]), int(colId[i]), 0] = thisRange[i]
    range_image[int(rowId[i]), int(colId[i])] = thisRange[i]
range_image = range_image.transpose()
range_image = np.flip(range_image, axis=0)
# print(range_image.shape)


##### 저장시 해당 코드는 모두 주석처리 #####
## range image 디스플레이
## 정면에서 볼때
# range_image = range_image.transpose()
# range_image = np.flip(range_image, axis=0)
# plt.imshow(X=range_image)
# plt.show()
##### 저장시 해당 코드는 모두 주석처리 #####




fw = open('test.rimg', 'wb')
fw.write(b'\x10\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00');
for i in range(0,range_image.shape[0]):      #1024
    for j in range(0,range_image.shape[1]):  #32
        L = range_image[i,j]
        s = np.float16(L).tobytes()
        fw.write(bytes([s[0],s[1]]));
fw.close()






