## Visual Description Description
* The datasets VSDv2 are available now.
<!-- <a href="https://github.com/unikcc/DiaASQ">
  <img src="https://img.shields.io/badge/DiaASQ-0.1-blue" alt="DiaASQ">
</a>
<a href="https://github.com/unikcc/DiaASQ" rel="nofollow">
  <img src="https://img.shields.io/badge/pytorch-1.8.1-green" alt="pytorch 1.8.1">
</a>
<a href="https://huggingface.co/docs/transformers/index" rel="nofollow">
  <img src="https://img.shields.io/badge/transformers-4.24.0-orange" alt="Transformers">
</a>
<a href="https://github.com/unikcc/DiaASQ/blob/master/LICENSE" rel="nofollow">
  <img src="https://img.shields.io/badge/LICENSE-MIT-cyan" alt="LICENSE">
</a> -->

This repository cotains code and data for our paper [Visual Spatial Description: Controlled Spatial-Oriented Image-to-Text Generation](https://arxiv.org/abs/2210.11109)

** Note **
Please go into [VLT5](https://github.com/j-min/VL-T5) and follow the README there for Pretrained Models and Feature Extraction.


## Setup
```bash
# Create python environment (optional)
conda create -n vsd python=3.10
source activate vsd

# Install python dependencies
pip install -r requirements.txt

```
安装[detectorn2==0.5](https://github.com/facebookresearch/detectron2/releases/tag/v0.5)

安装[language-evaluation](https://github.com/j-min/language-evaluation)

## Code structure
```bash
# Store images, features, and annotations
./datasets

# Image feature extraction
./feature_extraction

# Train VL-T5
./VL-T5/
    src/
        modeling_t5.py modeling_bart.py                       <= VL-T5/VL-BART model classes
        caption_sp.py, vrd_caption.py                         <= fine-tuning
        param.py                                              <= (argparse) configuration
        tokenization.py                                       <= custom tokenizer
        utils.py, dist_utils.py                               <= utility functions
    snap/                                                     <= store weight checkpoints
```


## Pretrained Models
- pretrained VL-BART and VL-T5 are provided by [VLT5](https://github.com/j-min/VL-T5)
- Download `snap/` from [Google Drive](https://drive.google.com/drive/folders/1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph?usp=sharing) or [feijiang](https://aistudio.baidu.com/datasetdetail/241378)
## Dataset
- This dataset is create from [VG](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) and [SpatialSense](https://zenodo.org/record/8104370) images by running
```bash
python feature_extraction/sp_proposal.py
or 
python feature_extraction/vg_proposal.py
```
- All you need to do is put all of the VG and SpatialSense images in a same folder
- the final .h5 file can be downloaded from [google]() and [feijiang]()
- put the .h5 file in the dataset folder and named it vsd_boxes36.h5
## train
- If your network can't connect to huggingface by url, you may need to download the [facebook/bart-base](https://huggingface.co/facebook/bart-base) and [t5-base](https://huggingface.co/t5-base) in your local.
```bash
bash train_b16.sh num_gpu
bash train_b80.sh num_gpu
# or if yout want to train one of them
bash test_batch_80/baseline_bart.sh num_gpu
# others are same
```
- The weights can be downloaded from [gdrive]() and [feijiang]()
## test
```bash
bash test_b16.sh 1 --use_golden
bash test_b80.sh 1 --use_golden
```
- The result files will save at test_16_res and test_80_res folder.

## result
### batch size 80
|  Model| BLEU-4  | METEOR  | ROUGE | CIDEr| SPICE| Acc|
|  ---- | ----  | ----  | ----  | ----  | ----  | ----  |
| VLBART | 54.52 |  43.10 | 78.79 | 482.64 | 68.95 | - |
| VLBART-end2end  | 52.90 |  42.15 | 77.60 | 469.65 | 67.64 | 52.22 |
| VLBART-end2end-golden  | 71.94 |  50.93 | 87.17 | 571.46 | 76.66 | 52.22 |
||
| VLT5  | 54.72 |  43.26 | 79.04 | 484.09 | 68.95 | - |
| VLT5-end2end  | 53.88 |  42.88 | 78.98 | 481.18 | 68.88 | 54.38 |
| VLT5-end2end-golden  | 72.24 |  51.21 | 87.92 | 576.20 | 76.95 | 54.38 |
| VLT5-end2end-onestep-test-false  | 53.71 |  42.76 | 78.49 | 476.68 | 68.31 | 52.41 |
| VLT5-end2end-onestep-test-true  | 53.82 |  42.75 | 78.58 | 477.96 | 68.58 | 52.41 |
| VLT5-end2end-onestep-golden  | 52.85 |  42.45 | 78.18 | 472.35 | 68.00 | 52.41 |
### batch size 16
|  Model| BLEU-4  | METEOR  | ROUGE | CIDEr| SPICE| Acc|
|  ---- | ----  | ----  | ----  | ----  | ----  | ----  |
| VLBART | 52.73 |  42.35 | 77.91 | 471.97 | 67.74 | - |
| VLBART-end2end  | 53.19 |  42.10 | 77.76 | 470.02 | 68.06 | 52.81 |
| VLBART-end2end-golden  | 71.77 |  50.75 | 87.28 | 568.66 | 76.80 | 52.81 |
||
| VLT5  | 54.44 |  43.03 | 78.82 | 484.02 | 68.92 | - |
| VLT5-end2end  | 54.76 |  43.10 | 79.08 | 481.46 | 68.58 | 53.27 |
| VLT5-end2end-golden  | 73.49 |  51.77 | 88.48 | 582.18 | 77.48 | 53.27 |
## Acknowledgement

This repo is adapted from [VLT5](https://github.com/j-min/VL-T5).

## 训练和测试注意事项

训练时分为3种

1、baseline，使用的input_id为'sub obj'

2、train(one_step_dec=True)，使用的input_id为'sub <extral_id_0> obj'

3、train(one_step_dec=False)，使用的input_id为'sub <extral_id_i> obj'其中i为方位词所对应的序号

论文所属参数使用为第三种训练方式

测试时分3种，

1、golden=False,one_step_dec=True，测试时输入input_id为'sub <extral_id_0> obj'

2、golden=False,one_step_dec=False，测试时输入input_id为'sub <extral_id_i> obj'，此处的i为模型预测方位词对应id

3、golden=True，测试时输入input_id为'sub <extral_id_i> obj'，此处i为数据集中标注方位词对应id