import glob
Low_config='configs/super_resolution/Low_Resolution.yaml'
High_config='configs/super_resolution/High_Resolution.yaml'

from torchpack.utils.config import configs

def check_redundancy(pcd_list, seq_list):
    seq_list['train']=[str(file.replace('\'','')) for file in seq_list['train']]
    seq_list['val']=[str(file.replace('\'','')) for file in seq_list['val']]
    seq_list['test']=[str(file.replace('\'','')) for file in seq_list['test']]

    train=[]
    val=[]
    test=[]
    for pcd in pcd_list:
        if list(set([pcd.split('/')[-3]]) & set(seq_list['train'])):
            train.append(pcd)
        elif list(set([pcd.split('/')[-3]]) & set(seq_list['val'])):
            val.append(pcd)
        elif list(set([pcd.split('/')[-3]]) & set(seq_list['test'])):
            test.append(pcd)
    print('total:',len(train)+len(val)+ len(test))
    print('train:', len(train))
    print('val:', len(val))
    print('test:', len(test))
if __name__== "__main__":
    #print(ouster_noise_save_path)
    print('32ch')
    configs.load(Low_config, recursive=True)
    check_redundancy(glob.glob('../../super_resolution/High_Resolution_bin/sequences/*/velodyne/*'),configs['seq'])
    print('128ch')
    configs.load(High_config, recursive=True)
    check_redundancy(glob.glob('../../super_resolution/High_Resolution_bin/sequences/*/velodyne/*'),configs['seq'])

