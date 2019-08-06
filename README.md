# PhaseCam3D

### [Project](http://yicheng.rice.edu/phasecam3d/) | [Video](https://www.youtube.com/watch?time_continue=751&v=CV4vEAjBv20) | [Paper](https://ieeexplore.ieee.org/document/8747330)

This repository contains tensorflow implementation for the ICCP2019 paper *PhaseCam3D — Learning Phase Masks for Passive Single View Depth Estimation* by [Yicheng Wu](http://yicheng.rice.edu), [Vivek Boominathan](https://vivekboominathan.com/), [Huaijin Chen](http://hc25.web.rice.edu/), [Aswin Sankaranarayanan](http://users.ece.cmu.edu/~saswin/) and [Ashok Veeraraghavan](http://computationalimaging.rice.edu/team/ashok-veeraraghavan/).

![PhaseCam3D framework](/figures/PhaseCam3D_framework.png)



## Installation
Clone this repo.
```bash
git clone https://github.com/YichengWu/PhaseCam3D
cd PhaseCam3D/
```
The code is developed using python 3.7.1 and tensorflow 1.13.0. The GPU we used is NVIDIA GTX 1080 Ti (11G). Change `batch_size` according if you run the code with different GPU memory.

## Dataset

The dataset is modified based on FlyingThings3D in [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). The pre-processed TFrecord files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/18b1CamTQd6wf2o3kxfL5aqWtWopIDuVG?usp=sharing). It contains 5077 training patches, 553 validation patches, and 419 test patches.

## Train

To train the entire framework, simply just run the following code
```
python depth_estimation.py
```
Inside the code, `results_dir` the output result directory, `DATA_PATH` is the directory of the download dataset. The learning rate of the optical layer and digital network can be set individually using `lr_optical` and `lr_digital`. Detailed instruction about choosing the learning rate can be found in Sec. IIID(d) in the paper.

### Logging

We use Tensorboard for logging training progress. Recommended: execute `tensorboard --logdir /path/to/save_dir --port 9001` and visit `localhost:9001` in the browser.

## Evaluation

Once the network is trained, the performance can be evaluated using the testing dataset. 
```
python depth_estimation.py
```
Change `results_dir` to the place you save your model. If you just want to see the performance of our best result, type `results_dir="./trained_framework/"`. Once the code is finished, a new folder called `test_all` will be created inside the model directory. It contains 400 scenes, and each one includes the clean image `sharp.png`, the coded image `blur.png`, the estimated disparity map `phiHat.png` and the ground truth disparity map `phiGT.png`. Sample images are shown below.

<p align="center">
  <img width="500" src="/figures/PhaseCam3D_sim_results.png">
</p>

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{wu2019phasecam3d,
  title={PhaseCam3D—Learning Phase Masks for Passive Single View Depth Estimation},
  author={Wu, Yicheng and Boominathan, Vivek and Chen, Huaijin and Sankaranarayanan, Aswin and Veeraraghavan, Ashok},
  booktitle={2019 IEEE International Conference on Computational Photography (ICCP)},
  pages={1--12},
  year={2019},
  organization={IEEE}
}
```
## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Yicheng Wu (yicheng.wu@rice.edu).
