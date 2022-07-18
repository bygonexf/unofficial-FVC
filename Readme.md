# Unofficial PyTorch implementation of [FVC: A New Framework towards Deep Video Compression in Feature Space](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_FVC_A_New_Framework_Towards_Deep_Video_Compression_in_Feature_CVPR_2021_paper.pdf)

This repository is built upon [CompressAI](https://github.com/InterDigitalInc/CompressAI) platform.

Please note that only the feature-sapce DCN part is implemented.

## TODO
- [] implement the multi-frame fusion part
- [] set GOP size in  args
- [] modify the visualization code of DCN offsets

## Setup

```
pip install -r requirements.txt
```

## Run

### Train

Run the command in the project root directory.

```bash
python examples/train_video.py -d ${DATA_PATH} --epochs 100 --batch-size 16 -m fvc --cuda --save
```

### Evaluation

Run the command in the project root directory.

```bash
python compressaidcn/utils/video/eval_model/__main__.py checkpoint ${DATA_PATH} ${OUTPUT_DIR} -a fvc -p ${MODEL_PATH} --keep_binaries -v
```

