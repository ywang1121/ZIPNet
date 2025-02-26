# HandDAGT: A Denoising Adaptive Graph Transformer for 3D Hand Pose Estimation

Wencan Cheng, Eun-ji Kim and Jong Hwan Ko

European Conference on Computer Vision (ECCV), 2024


## Prerequisities
Our model is trained and tested under:
* Python 3.6.9
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 1.9.0)
* scipy
* tqdm
* Pillow
* yaml
* json
* cv2
* pycocotools

1. Prepare dataset 

    please download the NYU Hand dataset

2. Install PointNet++ CUDA operations

    follow the instructions in the './pointnet2' for installation 

3. Evaluate

    set the "--dataset_path" paramter in the ```test_nyu.sh ``` as the path saved the generated testing set

    execute ``` sh test_nyu.sh```

    we provided the pre-trained models ('./pretrained_model/nyu_handdagt_3stacks/best_model.pth') for NYU

4. If a new training process is needed, please execute the following instructions after step 1 and 2 are completed

    set the "--dataset_path" paramter in the ```train_nyu.sh ``` as the path saved the generated traning and testing set respectively

    execute ``` sh train_nyu.sh```


## Acknowledgement

We thank [repo](https://github.com/PengfeiRen96/IPNet) for the image-point cloud framework.
