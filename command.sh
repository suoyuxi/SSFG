CUDA_VISIBLE_DEVICES=1 python tools/train.py /workspace/mmrotate/configs/csar/mcanet_1x_sl_oc.py --work-dir /workspace/mmrotate/workdir/20241203/sl
CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py /workspace/mmrotate/configs/csar/mcanet_1x_sl_oc.py --work-dir /workspace/mmrotate/workdir/20241206/sl_pearson 2>&1

CUDA_VISIBLE_DEVICES=7 nohup python pretrain.py 2>&1

CUDA_VISIBLE_DEVICES=1 python tools/test.py "/workspace/syx/workdir/MCA/SL/20241125/mca_dual_backbone/mcanet_1x_sl_oc.py" "/workspace/syx/workdir/MCA/SL/20241125/mca_dual_backbone/epoch_12.pth" --eval mAP

CUDA_VISIBLE_DEVICES=1 nohup python tools/train.py /workspace/mmrotate/configs/csar/mcanet_1x_fsi_oc.py --work-dir /workspace/mmrotate/workdir/20250105/fsi_pearson 2>&1

/workspace/mmrotate/workdir/20241219/sl/ (good)
/workspace/mmrotate/workdir/20241206/sl_pearson/ (good)

CUDA_VISIBLE_DEVICES=6 python tools/train.py /workspace/mmrotate/configs/csar/mcanet_1x_sl_oc.py --work-dir /workspace/mmrotate/workdir/20250105/sl_pearson
CUDA_VISIBLE_DEVICES=7 python pretrain.py

python3 tools/test.py '/workspace/mmrotate/workdir/sl_pearson/mcanet_1x_sl_oc.py' '/workspace/mmrotate/workdir/sl_pearson/sl_pearson.pth' --eval mAP



docker run -it --gpus all --name=mmrotate -p 12138:22 -v D:\research:/workspace youweigq/py37-pyth170-cuda101-cudnn8:v4.0 /bin/bash
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu125/torch2.4.0/index.html