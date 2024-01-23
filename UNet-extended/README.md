# ThermUnet (Deep Learning Approaches for Thermographic Imaging)
[Project](https://zenodo.org/record/5205460#.YRo5Co4zYuV/) **|** [Paper](https://aip.scitation.org/doi/full/10.1063/5.0020404/)


[Péter Kovács](https://www.researchgate.net/profile/Peter_Kovacs12),
[Bernhard Lehner](https://www.researchgate.net/profile/Bernhard_Lehner),
[Gregor Thummerer](https://www.researchgate.net/profile/Gregor_Thummerer),
[Günther Mayr](https://www.researchgate.net/profile/Guenther_Mayr),
[Peter Burgholzer](https://www.researchgate.net/profile/Peter_Burgholzer),
and 
[Mario Huemer](https://www.researchgate.net/profile/Mario_Huemer/)

Deep Learning Approaches for Thermographic Imaging, Journal of Applied Physics - Photothermics, 2020

## Abstract
In this paper, we investigate two deep learning approaches to recovering initial temperature profiles from thermographic
images in non-destructive material testing. First, we trained a deep neural network (DNN) in an end-to-end fashion
by directly feeding the surface temperature measurements to the DNN. Second, we turned the surface temperature
measurements into virtual waves (a recently developed concept in thermography), which we then fed to the DNN. To
demonstrate the effectiveness of these methods, we implemented a data generator and created a dataset comprising a
total of 100,000 simulated temperature measurement images. With the objective of determining a suitable baseline,
we investigated several state-of-the-art model-based reconstruction methods, including Abel transformation, curvelet
denoising, and time- and frequency-domain synthetic aperture focusing techniques (SAFT). Additionally, a physical
phantom was created to support evaluation on completely unseen real-world data.
The results of several experiments suggest that both the end-to-end and the hybrid approach outperformed the baseline in terms of reconstruction accuracy. The end-to-end approach required the least amount of domain knowledge
and was the most computationally efficient. The hybrid approach required extensive domain knowledge and was more
computationally expensive than the end-to-end approach. However, the virtual waves served as meaningful features
that convert the complex task of the end-to-end reconstruction into a less demanding undertaking. This in turn yielded
better reconstructions with the same number of training samples compared to the end-to-end approach. Additionally,
it allowed more compact network architectures and use of prior knowledge, such as sparsity and non-negativity. The
proposed method is suitable for non-destructive testing in 2D where the amplitudes along the objects are considered to
be constant (e.g. for metallic wires). To encourage the development of other deep-learning-based reconstruction techniques, we release both the synthetic and the real-world datasets along with the implementation of the deep learning
methods to the research community.


## Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Inference on Pre-trained Models (Synthetic Data)](#inference-synthetic)
1. [Inference on Pre-trained Models (Real-world Data)](#inference-real)
1. [Generating Fig.6 and Fig.7](#fig6)
1. [Generating Fig.9](#fig9)
1. [Training New Models for the End-to-End Approach](#train-e2e)
1. [Training New Models for the Hybrid Approach](#train-hybrid)
1. [Inference on New Models](#inference-custom)
1. [Downloading Raw Real-World Measurements](#raw-measurements)
1. [Uncertainty Estimation](#uncertainty-estimation)






## Introduction <a name="introduction"></a>
We propose two approaches to tackling thermal reconstruction that build on the same architectures to allow direct comparison.
First, we trained deep neural networks in an end-to-end fashion.
That is, we directly fed the surface temperature data to the network.
Second, we utilized the virtual wave concept as a feature extraction step.
In this case, we fed the resulting mid-level representation to the neural networks.



## Citation <a name="citation"></a>
If you find the code and datasets useful in your research, please cite:

    @inproceedings{icasspNDT,
      title={A Hybrid Approach for Thermographic Imaging With Deep Learning},
      author={Kov\'acs, P. and Lehner, B. and Thummerer, G. and G\"unther, M. and Burgholzer, P. and Huemer, M.},
      booktitle={Proceedings of the 45th IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      pages={4277--4281},
      year={2020},
    }
    
    @article{kovac2020jap,
         title={Deep Learning Approaches for Thermographic Imaging},
         author={Kov\'acs, P. and Lehner, B. and Thummerer, G. and G\"unther, M. and Burgholzer, P. and Huemer, M.},
         journal={Journal of Applied Physics},
         pages={155103--16},
         year={2020}
         volume={128},
         number={15},
         year={2020},
         publisher={AIP Publishing LLC}
    }

    @inproceedings{lehner2020uncertainty,
        title={Uncertainty estimation for non-destructive detection of material defects with u-nets},
        author={Lehner, B and Gallien, T},
        booktitle={Proceedings of the 2nd International Conference on Advances in Signal Processing and Artificial Intelligence (ASPAI)},
        year={2020}
    }    
    

## Requirements and Dependencies <a name="requirements-and-dependencies"></a>
- Ubuntu (we test with 19.10) or Windows 10 Pro (we test with Version 1903 for x64)
- Python (we test with Python = 3.6.8 in Anaconda3 = 4.8.3)
- Cuda & Cudnn (we test with Cuda = 10.1 and Cudnn = 7.6.0)
- PyTorch (we test with PyTorch = 1.0.1)
- NVIDIA GPU (we test with Titan X (Pascal) with compute = 6.1)

## Installation <a name="installation"></a>
### Windows
Setup [Anaconda](https://www.anaconda.com/) environment:

    $ conda env create -f environment_win.yml
    $ conda activate thermunet

### Ubuntu
Setup [Anaconda](https://www.anaconda.com/) environment:

    $ conda env create -f environment.yml
    $ conda activate thermunet

Download repository:

    $ git clone https://git.silicon-austria.com/pub/confine/ThermUNet.git

## Data
Download data.zip and **unpack inside the Project Folder** (ThermUNet)  
[here](https://zenodo.org/record/5205460#.YRo5Co4zYuV)

You should have the following folder structure inside your project folder. Please make sure that this is in order, the scripts rely on this!

    ./data/baselines/test/SNR_*: baseline results used in the paper
    
    ./data/end2end/models/cmp/*k: pretrained compact models from the end-to-end approach, trained with different number of samples from 10,000 to 80,0000.
    ./data/end2end/models/lrg/*k: pretrained large models from the end-to-end approach, trained with different number of samples from 10,000 to 80,0000.
    ./data/end2end/realworld/Deg_*: real-world measurements for the end-to-end approach taken from several degrees of rotation.
    ./data/end2end/test/SNR_*: 1,000 examples used as unseen data for the end-to-end approach, computed for 10 SNRs each. All in all, 10,000 examples.
    ./data/end2end/train/SNR_*: 10,000 examples for the end-to-end approach, computed for 10 SNRs each. All in all, 100,000 examples.

    ./data/hybrid/models/cmp/*k: pretrained compact models from the hybrid approach, trained with different number of samples from 10,000 to 80,0000.
    ./data/hybrid/models/lrg/*k: pretrained large models from the hybrid approach, trained with different number of samples from 10,000 to 80,0000.
    ./data/hybrid/realworld/Deg_*: real-world measurements for the hybrid approach taken from several degrees of rotation.
    ./data/hybrid/test/SNR_*: 1,000 examples used as unseen data for the hybrid approach, computed for 10 SNRs each. All in all, 10,000 examples.
    ./data/hybrid/train/SNR_*: 10,000 examples for the hybrid approach, computed for 10 SNRs each. All in all, 100,000 examples.

    ./data/masks/test: 1,000 ground truths
    ./data/masks/train: 10,000 ground truths



## Inference on Pre-trained Models (Synthetic Data) <a name="inference-synthetic"></a>
Run the [jupyter](https://jupyter.org/) notebook `inference_full.ipynb` to generate all results from the synthetic data set as used in the paper.


## Inference on Pre-trained Models (Real-world Data) <a name="inference-real"></a>
Run the [jupyter](https://jupyter.org/) notebook `inference_realworld.ipynb` to generate all results from the real-world data set as used in the paper.


## Generating Fig.6 and Fig.7 <a name="fig6"></a>
Run the [jupyter](https://jupyter.org/) notebook `plot_fig6_fig7.ipynb` to generate the figures as used in the paper.
In case you are having problems, there is also a notebook `plot_fig6_fig7_reference.ipynb` that additionally contains the outputs of the cells of the notebook as a reference.


## Generating Fig.9 <a name="fig9"></a>
Run the [jupyter](https://jupyter.org/) notebook `plot_fig9.ipynb` to generate the figures as used in the paper.
Please be aware that you need to run `inference_full.ipynb` first in order to yield the results that this script needs.  
In case you are having problems, there is also a notebook `plot_fig9_reference.ipynb` that additionally contains the outputs of the cells of the notebook as a reference.


## Training New Models for the End-to-End Approach <a name="train-e2e"></a>
Run the [jupyter](https://jupyter.org/) notebook `train_end2end.ipynb` to generate all results from the synthetic data set as used in the paper.
There are options to choose either the compact or the large architecture and the size of the training data, explained in the comments inside the notebook.
In case you are having problems, there is also a notebook `train_end2end_reference.ipynb` that additionally contains the outputs of the cells of the notebook as a reference.


## Training New Models for the Hybrid Approach <a name="train-hybrid"></a>
Run the [jupyter](https://jupyter.org/) notebook `train_hybrid.ipynb` to generate all results from the synthetic data set as used in the paper.
There are options to choose either the compact or the large architecture and the size of the training data, explained as comments inside the notebook.
In case you are having problems, there is also a notebook `train_hybrid_reference.ipynb` that additionally contains the outputs of the cells of the notebook as a reference.


## Inference on New Models <a name="inference-custom"></a>
Have a look at the [jupyter](https://jupyter.org/) notebook `inference_custom.ipynb` for an example.

## Downloading Raw Real-World Measurements <a name="raw-measurements"></a>
In case you want to conduct more experiments, the raw measurements (RawData.zip) can be downloaded [here](https://zenodo.org/record/5205460#.YRo5Co4zYuV)

## Uncertainty Estimation <a name="uncertainty-estimation"></a>
This is an extension of our work that was presented in the ASPAI 2020 paper.
In this paper, we propose two computationally very cheap methods to estimate the uncertainty of such u-net predictions.
We demonstrate the efficacy of our method through the utilization as a loss proxy for rejection above a specific loss.
For this, we use two models that differ in training data and compare with an ensemble based method.

### Downloading Additional Data
In case you want to reproduce the Figures from the paper, the raw measurements (ASPAI.zip) can be downloaded [here](https://zenodo.org/record/5205460#.YRo5Co4zYuV)

Please make sure the data is located after extraction in 

    ./data/ASPAI

### Interface demo
A short introdution on how to use the interfaces is given in the [jupyter](https://jupyter.org/) notebook `ASPAI_uncertainty_demo.ipynb`.
In case you are having problems, there is also a notebook `ASPAI_uncertainty_demo_reference.ipynb` that additionally contains the outputs of the cells of the notebook as a reference.


### Generating Fig.1 and Fig.3
Run the [jupyter](https://jupyter.org/) notebook `ASPAI_plot_fig1_fig3.ipynb`.


## Contact

[Péter Kovács](mailto:kovika@inf.elte.hu),
[Bernhard Lehner](mailto:bernhard.lehner@silicon-austria.com)

## License
See [MIT License](https://git.silicon-austria.com/pub/confine/ThermUNet/master/LICENSE)
