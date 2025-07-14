
# Intro

This repo is only for training pipeline and implementation. Dataset is not provided in this repo.

# Training Data preparation

We use AlphaPose to generate 2D ref pose and MotionBERT to generate 3D ref poses.

## Inference Camera 2D Pose

```
git clone and open the directory to project AlphaPose
follow the project to install corresponsing env

conda activate alphapose

python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir ${your image paths}  --save_img

```

## Check results on folder

```
cd /your_path_to_the/AlphaPose/examples/res

```


## Inference Camera 3D Pose

<!-- 根据需求 修改路径  -->
```
cd  /your_path_to_the/MotionBERT

conda activate motionbert

python infer_wild.py  --vid_path /your_path_to_the/infra_cam  --json_path /your_path_to_the/AlphaPose/examples/res/alphapose-results.json --out_path /your_path_to_the/test_out

```


## Check results on folder

```
cd {--out_path /your_path_to_the/test_out}

```


## split the dataset 

Generate the npy file to finish the data prepration

```
pred_data = np.load("/Users/geeklee/Projects/EMG/1205_ys_fw_1_X3D.npy")  
print(pred_data.shape)
<!-- 数据切分 example -->

split_data = pred_data[0:200, : , :]

```

# Prepare the Multi-Modality Data 

refer to the script in read_data.py and  data_preproc_base.py to align the timestamp and data