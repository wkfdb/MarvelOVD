# MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection
Official implementation of MarvelOVD in ECCV 2024.

## Installation

Our project is developed on [Detectron2](https://github.com/facebookresearch/detectron2).
Please follow the official installation [instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).


## Data Preparation

Download the [COCO dataset](https://cocodataset.org/#home), and put it in the `datasets/` directory.
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

Please refer to [pseudo label generation instruction](https://github.com/wkfdb/MarvelOVD/blob/main/PL_GENERATION.md) to generate offline pseudo-labels.

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
We train the model under regulr data augmentations (no Large Scale Jittering), without extra GPU memory occupation.

Training command
```bash 
python train_net.py --config configs/coco_openvoc_LSJ.yaml  --num-gpus=4
```
