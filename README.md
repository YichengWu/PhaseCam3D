# PhaseCam3D

### [Project](http://yicheng.rice.edu/phasecam3d/) | [Video](https://www.youtube.com/watch?time_continue=751&v=CV4vEAjBv20) | [Paper](https://ieeexplore.ieee.org/document/8747330)

This repository contains tensorflow implementation for the paper *PhaseCam3D — Learning Phase Masks for Passive Single View Depth Estimation* by [Yicheng Wu](http://yicheng.rice.edu), [Vivek Boominathan](https://vivekboominathan.com/), [Huaijin Chen](http://hc25.web.rice.edu/), [Aswin Sankaranarayanan](http://users.ece.cmu.edu/~saswin/) and [Ashok Veeraraghavan](http://computationalimaging.rice.edu/team/ashok-veeraraghavan/).

![Experimental results using PhaseCam3D](/figures/PhaseCam3D_exp_results.png)

## Installation
Clone this repo.
```bash
git clone https://github.com/YichengWu/PhaseCam3D
cd PhaseCam3D/
```
The code is developed using python 3.7.1 and tensorflow 1.13.0.

## Dataset

The dataset is modified based on FlyingThings3D in [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). The pre-processed TFrecord files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/18b1CamTQd6wf2o3kxfL5aqWtWopIDuVG?usp=sharing). It contains 5077 training patches, 553 validation patches, and 419 test patches.

## Train

## Evaluation

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
