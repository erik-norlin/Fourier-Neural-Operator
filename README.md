# Fourier-Neural-Operator

### Aim of this project ### 
This has been a smaller scientific group project about investigating the resolution invariance of a Fourier neural operator (FNO) when predicting Navier-Stokes dynamics. The project is based on the paper [*Fourier Neural Operator For Parametric Partial Differential Equations*](https://arxiv.org/pdf/2010.08895.pdf) (Zongyi Li et al., 2021).



The original code for this paper can be found on [ixScience's GitHub repository.](https://github.com/ixScience/fourier_neural_operator/tree/master)

### What the project does ### 
The program allows one to ...

The project consists of a couple different scripts that lets you
* Generate fluid flows to both .npy and .mat files (generate_NavierStokes_2d.ipynb)
* Convert .mat files to .npy (convert_dataset.ipynb)
* Train and evaluate an FNO (main.ipynb)
* Plot and create gifs of true/predicted fluid flows (plot.ipynb)
  
### How to run the project ### 



In every script there is a variable "$pathToProject" that is used to find and save files, assign this variable to the path of where you downloaded this repository. In some scripts there is a variable "gpu", set this to true if CUDA is available for faster computation. To train an FNO 


![](https://github.com/erik-norlin/CARMEN/blob/master/Plots/Qps/Qps_forest/forest_t%3D360000.jpeg?raw=true)

### Creators of this project ### 
* Gustav Burman
* Karl Lundgren
* Erik Norlin
* Mattias Wiklund

For more details, please read the project report.
