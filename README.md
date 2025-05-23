# ZIPNet: Zoom Image-Point Network for 3D Hand Pose Estimation
Network Architecture
![image](vis/ZIPNet.png)

## Prerequisities
Our model is trained and tested under:
* Python 3.10.15
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 2.1.1+CUDA12.1)
* scipy
* tqdm
* Pillow
* yaml
* json
* cv2
* pycocotools
* causal_conv1d == 1.0.0
* mamba_ssm == 1.0.1

1. Prepare dataset 

    please download the [NYU](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm) Hand dataset

2. Install PointNet++ CUDA operations

    follow the instructions in the './pointnet2' for installation 

3. Evaluate

    set the "--dataset_path" paramter in the ```test_nyu.sh ``` as the path saved the generated testing set

    execute ``` sh test_nyu.sh```

    we provided the pre-trained models ('./pretrained_model/nyu_handdagt_3stacks/best_model.pth') for NYU

4. If a new training process is needed, please execute the following instructions after step 1 and 2 are completed

    set the "--dataset_path" paramter in the ```train_nyu.sh ``` as the path saved the generated traning and testing set respectively

    execute ``` sh train_nyu.sh```

## Visualization
![image](vis/DAT.png)
![image](vis/real-sample.png)
![image](vis/dexycb_vis.png)

## Acknowledgement

We thank [repo](https://github.com/cwc1260/HandDAGT) for the HandDAGT framework.
