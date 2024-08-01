from typing import Any, Dict

import numpy as np
import torch
from torchpack import distributed as dist
from torchpack.callbacks.callback import Callback
import csv

__all__ = ['MeanIoU']

import csv

class MeanIoU(Callback):

    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 eval_mode: bool,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        #
        self.eval_mode=eval_mode 
        self.TP=[]
        self.FP=[]
        self.FN=[]
        self.TN=[]
        self.total=[]
        self.count=0
        for i in range(self.num_classes):
            self.total.append(0)
            self.TP.append(0)
            self.FP.append(0)
            self.FN.append(0)
            self.TN.append(0)
    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        f=open('High_Resolution.csv','a', newline='')
        wr=csv.writer(f)
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]
        self.count+=1
        tmp=[]
        for i in range(self.num_classes):
            self.total[i]+=int(outputs.shape[0])
#             self.TP[i]+=(torch.sum((targets == i) & (outputs == targets)).item())/outputs.shape[0]
#             self.FN[i]+=(torch.sum(targets == i).item()-(torch.sum((targets == i) & (outputs == targets)).item()))/outputs.shape[0]
#             self.FP[i]+=(torch.sum(outputs == i).item()-(torch.sum((targets == i)& (outputs == targets)).item()))/outputs.shape[0]
#             self.TN[i]+=1- (torch.sum(targets == i).item()+torch.sum(outputs == i).item()+torch.sum(targets == i).item()-torch.sum((targets == i) & (outputs == targets)).item())/outputs.shape[0]
            
            

            if outputs.shape[0]!=0:
                TP=(torch.sum((targets == i) & (outputs == targets)).item())/outputs.shape[0]
                FN=(torch.sum((targets == i) & (outputs != targets)).item())/outputs.shape[0]
                FP=(torch.sum((outputs == i)& (outputs != targets)).item())/outputs.shape[0]
                TN=(torch.sum((outputs!=i)&(targets!=i)).item())/outputs.shape[0]
                self.TP[i]+=TP
                self.FN[i]+=FN
                self.FP[i]+=FP
                self.TN[i]+=TN

            else:
                self.c[i]+=1
                #self.TP[i]+=(torch.sum((targets == i) & (outputs == targets)).item())/outputs.shape[0]
                #self.FN[i]+=(torch.sum((targets == i) & (outputs != targets)).item())/outputs.shape[0]
                #self.FP[i]+=(torch.sum((outputs == i)& (outputs != targets)).item())/outputs.shape[0]
                #self.TN[i]+=(torch.sum((outputs!=i)&(targets!=i)).item())/outputs.shape[0]


            total_seen= torch.sum(targets == i).item()
            total_correct= torch.sum((targets == i)
                                               & (outputs == targets)).item()
            total_positive= torch.sum(outputs == i).item()

            if (total_seen + total_positive - total_correct)!=0:
                iou=total_correct /( total_seen + total_positive - total_correct)
            else:
                iou=1
            self.total_seen[i] += total_seen
            self.total_correct[i] += total_correct
            self.total_positive[i] += total_positive


            if self.eval_mode and outputs.shape[0]!=0: #evaluate mode
                if i==0:
                    cls_name='car'
                elif i==1:
                    cls_name='two_wheel_vehicle'
                elif i==2:
                    cls_name='pedestrian'
                elif i==3:
                    cls_name='road'
                elif i==4:
                    cls_name='sidewalk'
                elif i==5:
                    cls_name='fence'
                elif i==6:
                    cls_name='structure'
                elif i==7:
                    cls_name='trunk'
                elif i==8:
                    cls_name='pole'
                elif i==9:
                    cls_name='traffic_sign'
                elif i==10:
                    cls_name='traffic_light'
                print(cls_name,'->',iou,' IoU[','TP:',TP,', FN:', FN, ', FP', FP,', TN:', TN,']')
                tmp.append(round(iou, 6))
            elif self.eval_mode and outputs.shape[0]==0:
                print(cls_name,'->',1,' IoU[','TP:',0.0,', FN:', 0.0, ', FP', 0.0,', TN:', 1.0,']')
                tmp.append(1)
        wr.writerow(tmp)
        f.close()


    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i],
                                                reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i],
                                                   reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i],
                                                    reduction='sum')

        ious = []
        for i in range(self.num_classes):

            print(i,':TP:',abs(self.TP[i]/self.count))
            print(i,':FP:',abs(self.FP[i]/self.count))
            print(i,':FN:',abs(self.FN[i]/self.count))
            print(i,':TN:',abs(self.TN[i]/self.count))

            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou)
        
        miou = np.mean(ious)

        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou * 100)
        else:
            print(miou,'mIoU','=avg(car:',ious[0],', two_wheel_vehicle:',ious[1],', pedestrian:',ious[2],', road:', ious[3],', sidewalk:', ious[4], ', fence:',ious[5],',structure:',ious[6],', trunk:',ious[7],', pole:', ious[8], ', traffic_sign:', ious[9], ', traffic_light:',ious[10])
