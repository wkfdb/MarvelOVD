# Psuedo Label Generation

This document describes how to generate psuedo labels for open vocabulary/zero-shot object detection.

**Prerequsite**: 
- Download the [COCO dataset](https://cocodataset.org/#home), and put it in the `datasets/` directory. 
- Download the pretrained class agnostic proposal generator [weight](https://drive.google.com/file/d/18CJMLiitP6pPY9cTTlhnBCdAj0cmF5T7/view?usp=drive_link) and put it as `./tools/mask_rcnn_R_50_FPN_1x_base_num1.pth`.
- Install [Pytorch]()(>=1.7), [Detectron2](), [CLIP](https://github.com/openai/CLIP), [COCO API](https://github.com/cocodataset/cocoapi), OpenCV, scipy and tqdm.


## Genearte PLs on COCO Training Set

To accelerate PL generation, we divide this process into two stages. 
In the first stage, we get CLIP scores for top 100 region proposals with multiple processes.
In the second stage, we generate PLs with pre-computed CLIP scores.
Additionally, to avoid data leakage, we should generate PLs for all images in the training set instead of those with novel objects.

**Stage 1**: Open the directory `tools/`.
```
cd ./tools
```
Run the following commands in different temrinals. The pre-computed CLIP scores will be stored in `--save_dir ./CLIP_scores_for_PLs`.
```
CUDA_VISIBLE_DEVICES=0 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 0 --end 30000
CUDA_VISIBLE_DEVICES=1 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 30000 --end 60000
CUDA_VISIBLE_DEVICES=2 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 60000 --end 90000
CUDA_VISIBLE_DEVICES=3 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 90000
```

**Stage 2**: Generate PL using pre-computed CLIP scores.
```
python gen_PLs_from_CLIP_scores.py --gt_json ../datasets/coco/annotations/open_voc/instances_train.json --clip_score_dir ./CLIP_scores_for_PLs --pl_save_file ../datasets/coco/annotations/openvoc/train_novel_candidate_0.5.json
```

**(Optional)**: (In default, this step is not necessary.) If no text embedding in json file, please add text embedding into the json file for training or evaluation. The postfix `_txtEmb` will be added in the input file name and results in the final file `train_novel_candidate_0.5_txtEmb.json`. 
```
python add_textEmb_cocojson.py ../datasets/coco/annotations/openvoc/train_novel_candidate_0.5.json
```



