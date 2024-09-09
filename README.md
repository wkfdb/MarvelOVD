# MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection
Official implementation of MarvelOVD in ECCV 2024.

## Installation

Our project is developed on [Detectron2](https://github.com/facebookresearch/detectron2).
Please follow the official installation [instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).


## Data Preparation

Download the [COCO dataset](https://cocodataset.org/#home), and put it in the `datasets/` directory.

Download VL-PLM [pre-generated pseudo-labeled data](https://drive.google.com/drive/folders/1hv9YZF0mGUQVCmDgEzGxYi9FtqIukNBd?usp=drive_link) and our generated [candidate pseudo-label data](https://drive.google.com/file/d/1s5GsYu4v1sk6Y-9CncORhsf_e-bzsEjq/view?usp=sharing), and put them in the `datasets/open_voc` directory.

Dataset are organized in the following way:
```bazaar
datasets/
    coco/
        annotations/
            instances_train2017.json
            instances_val2017.json
            open_voc/
                instances_eval.json
                instances_train.json
        images/
            train2017/
                000000000009.jpg
                000000000025.jpg
                ...
            val2017/
                000000000776.jpg
                000000000139.jpg
                ...
        
```

## Pseudo label generation

MarvelOVD dynamically learns open-vocabulary knowledge from offline-generated pseudo-labels under the guidance from the online training detector.

If necessary, please refer to [pseudo label generation instruction](https://github.com/wkfdb/MarvelOVD/blob/main/PL_GENERATION.md) to generate offline pseudo-labels.

## Results on OVD-COCO
Mask R-CNN:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->

<th valign="bottom">Novel AP</th>
<th valign="bottom">Base AP</th>
<th valign="bottom">Overall AP</th>
<!-- TABLE BODY -->
<!-- ROW: with LSJ -->
 <tr>
<td align="center">38.9</td>
<td align="center">56.4</td>
<td align="center">51.8</td>
</tr>

</tbody></table>

## Training
We train the model under regular data augmentations (no Large Scale Jittering), without extra GPU memory occupation. (Runing on 4 GPUs with 24G Memory per GPU)

Training command
```bash 
python train_net.py --config configs/coco_ssod.yaml  --num-gpus=4
```


## Notes
The code is highly borrowed from [VL_PLM](https://github.com/xiaofeng94/VL-PLM), big thanks for the open-source commuity.
Questions and Issues, please contract wangk229@mail2.sysu.edu.cn
