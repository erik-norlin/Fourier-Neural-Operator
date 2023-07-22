# Fourier-Neural-Operator

### About this project ###
This has been a smaller scientific research project about investigating resolution invariance of a Fourier neural operator (FNO) when predicting fluid flows. The project is based on the paper [*Fourier Neural Operator For Parametric Partial Differential Equations*](https://arxiv.org/pdf/2010.08895.pdf) (Zongyi Li et al., 2021) and the original code for this paper, that our implementation is heavily influenced by, can be found on [ixScience's GitHub page](https://github.com/ixScience/fourier_neural_operator/tree/master).

### Aim of this project ### 
The objective in this project was to investigate the resolution invariance of an FNO predicting fluid flows in two spatial dimensions given an initial sequence of discretizations described by the Navier−Stokes equations. To this end, an FNO was implemented and trained on a spatial resolution of 32 × 32, with the physical time 1 s between each frame. The FNO was then evaluated on a set of different resolutions, varying both spatially and temporally. Lastly, the spatial resolution invariance of the FNO was evaluated by comparing a high-resolution prediction with an upsampling of a low-resolution prediction using bicubic interpolation.

### Results ###
We found that the FNO performed exceptionally well at predicting fluid flows of higher spatial resolution than it was originally trained on. However, this was not the case for predicting higher temporal resolution because the error grew very quickly. 

<p align="center">
  <img src="https://github.com/erik-norlin/Fourier-Neural-Operator/blob/main/src/fno/output-flows/1024x1024/1024x1024_pred_true.gif">
  <i> Prediction and ground truth of a flow (vorticity) with spatial resolution 1024x1024 and temporal resolution of 1 s.</i>
</p>

Comparing our trained FNO with a bicubic interpolation it shows that our trained FNO performs better at upsampling spatial resolution every time.

<p align="center">
  <img src="https://github.com/erik-norlin/Fourier-Neural-Operator/blob/main/src/fno/output-flows/1024x1024/1024x1024_bicubic_fno_comp.gif">
  <i> Upsampling a flow (vorticity) with spatial resolution 1024x1024 and temporal resolution of 1 s using bicubic interpolation and a trained FNO respectively.</i>
</p>

<p align="center">
  <img src="https://github.com/erik-norlin/Fourier-Neural-Operator/blob/main/report/report_images/fno_bicube_comp_confint.png">
  <p><i> MAE from upsampling flows at different spatial resolutions using FNO and bicubic interpolation separately. The FNO performs better than bicubic interpolation at every resolution that was tested.</i></p>
</p>

## Implementation ##

### What the project does ###
With this implementation one can generate fluid flows of desired spatial and temporal resolution and then train and evaluate an FNO. The implementation purely as it is allows one to predict how fluid flow evolves over time for different spatial resolutions using our trained FNO.

More specifically, the project comes with four scripts that allows one to
* Generate fluid flows of either .npy or .mat files (generate_NavierStokes_2d.ipynb)
* Convert .mat files to .npy files (convert_dataset.ipynb)
* Train and evaluate an FNO (main.ipynb)
* Plot and create gifs of true and predicted fluid flows (plot.ipynb)
  
### How to run the project ### 

The scripts are Python Jupyter Notebooks that one has to run through. In each of them there is a variable *$pathToProject* that is used to find and save files, assign this variable to the path of where you choose to download this repository ending with */Fourier-Neural-Operator*. In some scripts there is a variable *gpu*, set this to true if CUDA is available for faster computation. To train and/or evaluate the FNO, simply run through the *main.ipynb* script.

### Datasets ###
Training data can either be found on [ixScience's Google Drive](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) or generated with *generate_NavierStokes_2d.ipynb*. If you choose to download and use any their datasets make sure to put it in the */src/data/datasets* directory. Small datasets of fluid flows of spatial resoltions 32x32 and 128x128 with temporal resolution 1 s already exists in the datasets directory. These are for evaluating the FNO.

### Requirements ###
As mentioned before, the scripts are created in Python Jupyter Notebook. All modules requried to run the scripts are named in *requirements.txt*.

### Creators of this project ### 
* Gustav Burman
* Karl Lundgren
* Erik Norlin
* Mattias Wiklund

For more details, please read the project report.
