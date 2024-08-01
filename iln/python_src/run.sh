#!/bin/bash
  
#SBATCH -J  4dsr                       # 작업 이름
#SBATCH -o  ./log/out.train_4dsr_0.%j          # stdout 출력 파일 이름 (%j는 %jobId로 확장됨)
#SBATCH -e  ./log/err.train_4dsr_0.%j          # stderr 출력 파일 이름 (%j는 %jobId로 확장됨)
#SBATCH -p  gpu-all                           # 큐 또는 파티션 이름
#SBATCH -t  7-00:00:00                        # 최대 실행 시간 (hh:mm:ss) - 1.5시간
#SBATCH -N 1                                # 노드 개수  
#SBATCH -n 1                                # ntasks 작업에서 사용할 총 task 수
#SBATCH --gres=gpu:4                        # 요청한 노드당 GPU 수
. /usr/share/modules/init/bash
source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
module  purge
module  load ohpc cuda/11.3

echo  $SLURM_SUBMIT_HOST
echo  $SLURM_JOB_NODELIST
echo  $SLURM_SUBMIT_DIR

echo  ### START ###

### cuda  test  ###

conda activate 4dsr

python train_models.py --config models/iln/config/iln_0d_4dsr.yaml
date  ; echo  ##### END #####
