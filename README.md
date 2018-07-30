# Image Algorithm for General Purpose


Image Algorithm is a clustering algorithm based [fast search and find of density peaks](http://science.sciencemag.org/content/344/6191/1492). 
Comparing with other popular clustering methods, such as DBSCAN,  one of the most prominent advantages of Image Algorithm is being highly parallelizable. This repository is an implementation of Image Algorithm for general purpose, supporting strong and easy GPU acceleration. 

For now, the implementation includes three backends: numpy, CUDA and OpenCL.

| backend | dependency | Support Platform | Support Device |
| :---: | :---: | :---: | :---: |
| [`numpy`](http://www.numpy.org) | None | Mac/Linux/Windows | CPU |
| [`CUDA`](https://en.wikipedia.org/wiki/CUDA) | pycuda | Linux | Only NVIDIA GPU |
| [`OpenCL`](https://en.wikipedia.org/wiki/OpenCL) | pyopencl | Mac | NVIDIA/AMD/Intel GPU, multi-core CPU |

For three backends, two kinds of data structure can be taken in. Flat list and KDBin. KDBins is based on hash map and memory reference to obtain nearest neighboring bins and points inside for each point. Performance test shows a strong acceleration in density calculation using KDBin data structure.

| supported data structure |  `rho` Calculation | `rhorank` and `nh` Calculation |
| :---: | :---: | :---: |
| [`numpy`](http://www.numpy.org) | list/kdbin | list/kdbin |
| [`CUDA`](https://en.wikipedia.org/wiki/CUDA) | list/kdbin | list |
| [`OpenCL`](https://en.wikipedia.org/wiki/OpenCL) | list/kdbin | list |
 
It has been tested that all three backends give the identical clustering results. Therefore users can feel free to choose whichever faster and easier for their purposes. Concerning speed performace, acceleration from CUDA/OpenCL may give an up to x20 speed up from CPU when dealing with more than a few thousands of data points. A preliminary speed test of three backends can be found [here](https://galleryziheng.wordpress.com/2017/12/08/gpu-acceleration-of-imaging-algorithm).


To do list
    
* [ ] Calculate `rhorank` in CUDA/OpenCL via 'Merge Sorted List' for parallel sorting.
* [ ] Support `nh` search in nearest neighboring bins in CUDA/OpenCL


## Installation 


```bash
pip install ImageAlgoKD
```

Regarding dependency, no dependency is required for numpy backend. And it usually does a good job dealing with small dataset and needs no extra packages. However, for users wanting to use GPU acceleration with either CUDA or OpenCL backend, extra dependency is required. 

```bash
# if want to use opencl backend
pip install pyopencl
# if want to use CUDA backend
pip install pycuda
```

## Quick Start
The primary usage of the module is the following
First of all, import ImageAlgo class for K-Dimension
```python
from ImageAlgoKD import *
```

Declare an instance of ImageAlgoKD with your algorithm parameters. Then give it the input data points.
```python
ia = ImageAlgoKD(MAXDISTANCE=20, KERNEL_R=1.0)
ia.setInputsPoints(Points(np.genfromtxt("../data/basic.csv",delimiter=',')))
```

Then run the clustering over input data points.
```python
ia.run("numpy")
# ia.run("opencl") or ia.run("cuda") if want run in parallel
```

In the end, the clustering result can be access by
```python
ia.points.clusterID
```

## Algorithm Parameters

|     Parameters     | Comments                                                       | Default Value |
|:------------------:|----------------------------------------------------------------|:-------------:|
|     MAXDISTANCE    | the separation distance of the point with highest density.     |      10.0     |
|      KERNEL_R      | 'd_c' in density calculation                                   |      1.0      |
|    KERNEL_R_NORM   | 'd_0' in density calculation                                   |      1.0      |
|   KERNEL_R_POWER   | 'k' in density calculation.                                    |      0.0      |
| DECISION_RHO_KAPPA | the ratio of density threshold of seeds to the highest density |      4.0      |
|    DECISION_NHD    | the separation threshold of seeds                              |      1.0      |
|   CONTINUITY_NHD   | the separation threshold of continuous clusters                |      1.0      |

where density is defined as

<p align=center><img width="25%" src=https://github.com/ZihengChen/ImageAlgorithm/blob/master/plots/density.png /></p> 

## Examples

### I. Basic

<p align=center><img width="60%" src=https://github.com/ZihengChen/ImageAlgorithm/blob/master/plots/basic.png   /></p>

Perform IA clustering on 1000 toy 2D points, sampled from two Gaussian Distrituion and noise. The toy data is in `data/basic.csv`, while the corresponding jupyter notebook can be found [here](/example/example_basic.ipynb) in `examples/`.

### II. MNIST
<p align=center> 
  <img width="40%" src=https://github.com/ZihengChen/ImageAlgorithm/blob/master/plots/mnist_decision.png /> 
  <img width="45%" src=https://github.com/ZihengChen/ImageAlgorithm/blob/master/plots/mnist.png />
</p> 

Perform IA clustering on 1000 MNIST 28x28 dimension points. The MNIST data is in `data/mnist.csv`, while the corresponding jupyter notebook can be found [here](/example/example_mnist.ipynb) in `examples/`.

### III. HGCal


<p align=center><img width="50%" src=https://github.com/ZihengChen/ImageAlgorithm/blob/master/plots/hgcal.png /></p> 

This is an event of 10 Pions with 300 GeV energy in CMS HGCal. A 3D interactive visualization can be found [here](https://plot.ly/~zihengchen/61/#/). In addition, for event with pile up, [here](https://plot.ly/%7Ezihengchen/18/#/) is an 300GeV pion with PU200 event. A PU200 event typically includes about 200k HGVCal reconstructed detector hits, which is input into IA clustering


