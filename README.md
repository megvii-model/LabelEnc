# LabelEnc: A New Intermediate Supervision Method for Object Detection

This is the LabelEnc implementation based on [FCOS](https://github.com/tianzhi0549/FCOS) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Our full paper is available at [https://arxiv.org/abs/2007.03282](https://arxiv.org/abs/2007.03282)

## Installation

See [INSTALL.md](INSTALL.md) for installation instructions.

Our implementation is based on [FCOS](https://github.com/tianzhi0549/FCOS) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). You may refer to them for installation instructions as well.

## Training

LabelEnc uses a two-step training pipeline. In step1, we acquire a Label Encoding Function. In step2, we train the final detection model.

### Configure COCO dataset

You will need to download the COCO dataset. We recommend to symlink the path to the coco dataset to `datasets/` as follows

```bash
# symlink the coco dataset
cd LabelEnc
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
```

### Training for Step1

Use the following commands to train LabelEnc Step1 with on 8 GPUs. Take retinanet_R-50-FPN for example:
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train_labelenc_step1.py \
    --config-file configs/retinanet/retinanet_R-50-FPN_1x.yaml \
    OUTPUT_DIR exps/step1/retinanet_R-50-FPN/
```
The Label Encoding Function will be saved to `exps/step1/retinanet_R-50-FPN/label_encoding_function.pth`.

Note that:
1. The shown AP is the precision of the auxiliary detection model in Step1 (denoted in the paper as "Step1 only" in Table 4).
2. We use 1x schedule in both Step1 and Step2. Use only `xxx_1x.yaml` when training LabelEnc, or change the schedule in the config file to [60000, 80000, 90000] yourself.

### Training for Step2

Use the following commands to train LabelEnc Step2 with `retinanet_R-50-FPN_1x.yaml` on 8 GPUs. Use `--labelenc` command to specify `label_encoding_function.pth` file. Take retinanet_R-50-FPN for example:
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train_labelenc_step2.py \
    --config-file configs/retinanet/retinanet_R-50-FPN_1x.yaml \
    --labelenc exps/step1/retinanet_R-50-FPN/label_encoding_function.pth \
    OUTPUT_DIR exps/step2/retinanet_R-50-FPN/
```

## Models

We provide the following trained LabelEnc models.

|Model|Baseline AP (minival)|LabelEnc AP (minival)|Link|
|-----|-----------|-----------|----|
|retinanet_R-50-FPN_1x|36.4|38.6|[download](https://drive.google.com/file/d/1b-P_OXltGaRucnI5fChNk3UCQwQyiZDQ/view?usp=sharing)|
|retinanet_R-101-FPN_2x|38.5|40.5|[download](https://drive.google.com/file/d/1hGcxmnJzCUym4KbPPSgH05paCuyWJX2H/view?usp=sharing)|
|retinanet_dcnv2_R-101-FPN_2x|41.5|42.9|[download](https://drive.google.com/file/d/1Tdav2tDP3hOMUhBEuax5SP5S9vudcY-d/view?usp=sharing)|
|fcos_R_50_FPN_1x|37.1|39.5|[download](https://drive.google.com/file/d/1GpexaS-W9pnxV-cR-IFScCI0_Gvkzczq/view?usp=sharing)|
|fcos_imprv_R_50_FPN_1x|38.7|41.1|[download](https://drive.google.com/file/d/17hbgrh_wyKunI9jECYfyXjgJ5tLNHIQS/view?usp=sharing)|
|fcos_imprv_R_101_FPN_2x|40.5|43.1|[download](https://drive.google.com/file/d/1RzFM9DgGcY47s8R5G7YlWYYOOVH7p1WM/view?usp=sharing)|
|fcos_imprv_dcnv2_R_101_FPN_2x|43.5|45.8|[download](https://drive.google.com/file/d/18eSyziTdKXlwOoaPPhV374atnmWgx13I/view?usp=sharing)|
|e2e_faster_rcnn_R_50_FPN_1x|36.8|38.6|[download](https://drive.google.com/file/d/1MsxibHuXrpRoGVZhDy2XokJc9oY_WUps/view?usp=sharing)|
|e2e_faster_rcnn_R_101_FPN_1x|39.1|41.0|[download](https://drive.google.com/file/d/15j_7ltquUYP-7OKT1HwgW2zMbZPVu6_A/view?usp=sharing)|
|e2e_faster_rcnn_dcnv2_R_101_FPN_2x|42.8|43.3|[download](https://drive.google.com/file/d/1t28rXVQIJ3XWX-LHqqknf_7QegROnZrI/view?usp=sharing)|

[1] `1x` and `2x` only notes the schedule for baseline models. LabelEnc models use `1x` in both steps.

[2] We use `R_50` models in step1 for all models above, no matter their backbone in step2. For example, `retinanet_dcnv2_R-101-FPN` uses `label_encoding_function.pth` from `retinanet_R-50-FPN`.

Links for step1 weights, i.e. `label_encoding_function.pth` are listed below.

|Model|Link|
|-----|----|
|retinanet_R-50-FPN|[download](https://drive.google.com/file/d/1qcAxuYVy2M_OdQDEIXzl4T5vAUmrJOWx/view?usp=sharing)|
|fcos_R_50_FPN|[download](https://drive.google.com/file/d/1WBkzsMakP5DudPT_am1giHxMrLY4LjQR/view?usp=sharing)|
|fcos_imprv_R_50_FPN|[download](https://drive.google.com/file/d/1dFuEL06qKbbsxzEK7RshX7OIrPDfmvDw/view?usp=sharing)|
|fcos_imprv_dcnv2_R_50_FPN|[download](https://drive.google.com/file/d/1VVlHElOsbxBavgHCiTTM5MJmL9TtonGV/view?usp=sharing)|
|e2e_faster_rcnn_R_50_FPN|[download](https://drive.google.com/file/d/1hnrbjAVPlWm4f2cxqH0e4mbvlqZF_zI2/view?usp=sharing)|

\* We train step1 models individually for `fcos_xxx`, `fcos_imprv_xxx` and `fcos_imprv_dcnv2_xxx` because they have different detection heads (`fcos_imprv_dcnv2` uses dcn in head in the official release).


## Citations
You may cite our work in your publications with the following BibTex codes:
```
@article{hao2020labelenc,
  title={LabelEnc: A New Intermediate Supervision Method for Object Detection},
  author={Hao, Miao and Liu, Yitao and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2007.03282},
  year={2020}
}
```
