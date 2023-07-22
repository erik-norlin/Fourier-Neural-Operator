# Fourier-Neural-Operator

### About this project ###
This has been a smaller scientific research project about investigating the resolution invariance of a Fourier neural operator (FNO) when predicting Navier-Stokes dynamics. The project is based on the paper [*Fourier Neural Operator For Parametric Partial Differential Equations*](https://arxiv.org/pdf/2010.08895.pdf) (Zongyi Li et al., 2021) and the original code for this paper, that our implementation is heavily influenced by, can be found in [ixScience's GitHub repository.](https://github.com/ixScience/fourier_neural_operator/tree/master) 

### Aim of this project ### 
The objective in this project was to investigate the resolution invariance of an FNO predicting fluid flows in two spatial dimensions given an initial sequence of discretizations described by the Navier−Stokes equations. To this end, an FNO was implemented and trained on a spatial resolution of 32 × 32, with the physical time 1 s between each frame. The FNO was then evaluated on a set of different resolutions, varying both spatially and temporally. Lastly, the spatial resolution invariance of the FNO was evaluated by comparing a high-resolution prediction with an upsampling of a low-resolution prediction using bicubic interpolation.

### Results ###
We found that the FNO performed exceptionally well at predicting fluid flows of higher spatial resolution than it was trained on. However, this was not the case for the temporal resolution because the error grew very quickly. 

Prediction | True |
:-------------------------:|:-------------------------:
![](?raw=true) | ![](?raw=true)

To compare our trained FNO with a more traditional upsampling method we set it side by side with a bicubic interpolation. It shows that our trained FNO performs better at upsampling spatial resolution.

Bicubic | FNO |
:-------------------------:|:-------------------------:
![](?raw=true) | ![](?raw=true)


## Implementation ##

### What the project does ###
With this implementation one can generate and/or convert data of fluid flows to then train and evaluate an FNO on desired spatial and temporal resolutions. The implementation purely as it is allows one to predict how fluid flow evolves over time for different spatial resolutions using our trained FNO.

More specifically, the project comes with four scripts that allows you to
* Generate fluid flows of both .npy and .mat files (generate_NavierStokes_2d.ipynb)
* Convert .mat files to .npy files (convert_dataset.ipynb)
* Train and evaluate an FNO (main.ipynb)
* Plot and create gifs of true and predicted fluid flows (plot.ipynb)
  
### How to run the project ### 

The scripts are Jupyter Notebooks that you have to run through. In each of them there is a variable *$pathToProject* that is used to find and save files, assign this variable to the path of where you downloaded this repository. In some scripts there is a variable *gpu*, set this to true if CUDA is available for faster computation. To train and evaluate an FNO, simply run through the main.ipynb script.


### Requirements ###
As mentioned before, the scripts are created in Jupyter Notebook. The following modules are requried to run the scripts.
* cprofile
* functools
* lightning
* mat73
* math
* matplotlib
* numpy
* operator
* pstats
* scipy
* snakeviz
* timeit
* torch
* tqdm
* wandb
* yaml

### Creators of this project ### 
* Gustav Burman
* Karl Lundgren
* Erik Norlin
* Mattias Wiklund

For more details, please read the project report.
