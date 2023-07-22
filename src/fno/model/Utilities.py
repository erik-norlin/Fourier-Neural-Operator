import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from functools import reduce
import operator
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib as mpl
import lightning as L
import cProfile
import pstats
import snakeviz.cli as cli
import scipy


def npyToTorchDataLoader(
    npyFilePath, npyExtraPath=None,
    distribution=[0.8, 0.15, 0.05],
    timeDistribution=[0.2, 0.8],
    batchSize=1,
    shuffleTraining=True,
    normalize=True, trainingMean=-1.1511e-08, trainingStd=0.4811, downSample=False, onlyTest=False, 
    gif=False, 
    temporal_pred=False,
    temp_res=None):
    
    tensorFormat = torch.tensor(np.load(npyFilePath))
    
    
    if gif: # Picking one flow and making it correct dimensions
        tensorFormat = tensorFormat[0]
        tensorFormat = tensorFormat[None,...]
        print(tensorFormat.shape)
    
    
    if downSample:
        tensorFormat = Interpolate(tensorFormat, toResolution=32)
        

    nSamples = len(tensorFormat)
    timeBreakIndex, timesToPredict = [int(tensorFormat.shape[3] * k) for k in timeDistribution]
    tensorFormatX, tensorFormatY = (
        tensorFormat[..., :timeBreakIndex],
        tensorFormat[..., timeBreakIndex:],
    )
    
    
    
    
    if temporal_pred:
        nSamples = len(tensorFormat)
        timeBreakIndex, timesToPredict = [int(tensorFormat.shape[3] * k) for k in timeDistribution]
        init_time_seq = int(1/temp_res)
        tensorFormat_t = tensorFormat[..., ::init_time_seq]
        pred_time_seq = int(init_time_seq*10)
        tensorFormatX, tensorFormatY = (
            tensorFormat_t[..., :10],
            tensorFormat[..., pred_time_seq:],   
        )
        print(tensorFormatX.shape, tensorFormatY.shape)
    
    
    
    
    
    del tensorFormat

    trainingIdx, validationIdx, _ = [int(nSamples * k) for k in distribution]
    trainingX, validationX, testX = (
        tensorFormatX[:trainingIdx, ...],
        tensorFormatX[trainingIdx : trainingIdx + validationIdx, ...],
        tensorFormatX[trainingIdx + validationIdx :, ...],
    )
    del tensorFormatX

    trainingY, validationY, testY = (
        tensorFormatY[:trainingIdx, ...],
        tensorFormatY[trainingIdx : trainingIdx + validationIdx, ...],
        tensorFormatY[trainingIdx + validationIdx :, ...],
    )
    del tensorFormatY
    
    if npyExtraPath is not None:
        extraTensorFormat = torch.tensor(np.load(npyExtraPath))
        extraTensorX, extraTensorY = (
        extraTensorFormat[..., :timeBreakIndex],
        extraTensorFormat[..., timeBreakIndex:])
        del extraTensorFormat

    def Normalize(tensor, mean, std):
        return (tensor - mean) / std

    if normalize:
        trainingX = Normalize(trainingX, trainingMean, trainingStd)
        validationX = Normalize(validationX, trainingMean, trainingStd)
        testX = Normalize(testX, trainingMean, trainingStd)
        if npyExtraPath is not None:
            extraTensorX = Normalize(extraTensorX, trainingMean, trainingStd)
        
    trainingX = trainingX.unsqueeze_(3).expand(-1, -1, -1, timesToPredict, -1)
    training = TensorDataset(trainingX, trainingY)
    del trainingX
    del trainingY
    
    validationX = validationX.unsqueeze_(3).expand(-1, -1, -1, timesToPredict, -1)
    validation = TensorDataset(validationX, validationY)
    del validationX
    del validationY

    testX = testX.unsqueeze_(3).expand(-1, -1, -1, timesToPredict, -1)
    test = TensorDataset(testX, testY)
    del testX
    del testY
    if onlyTest:
        testDataLoader = DataLoader(
            test, batch_size=batchSize, shuffle=False, num_workers=1
        )
        del test
        return testDataLoader
        
    trainingDataLoader = DataLoader(
        training, batch_size=batchSize, shuffle=shuffleTraining, num_workers=12
    )
    del training

    validationDataLoader = DataLoader(
        validation, batch_size=batchSize, shuffle=False, num_workers=12
    )
    del validation

    testDataLoader = DataLoader(
        test, batch_size=5, shuffle=False, num_workers=12
    )
    del test
    
    return trainingDataLoader, validationDataLoader, testDataLoader


def CreateGif(data, filename, print_res, cmin, cmax, colormap='turbo', milliseconds_per_frame=100, title = ''):  # Takes tensor data with X, Y, T only dimensions > 1
    if len(data[0]) != 0:
        data = data[0][None,:,:,:]
    data = data.squeeze_()
    data = data.detach().numpy()
    fig = plt.figure(figsize=(12,12))
    
    im = plt.imshow(data[:, :, 0],    # display first slice
                    animated=True,
                    cmap=colormap,               # color mapping
                    vmin=cmin, #np.min(data), # lowest value in numpy_3d_array
                    vmax=cmax)#np.max(data)) # highest value in numpy_3d_array
    if print_res == 'true':
        # norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        plt.clim(vmin=cmin, vmax=cmax)
        colorbar = plt.colorbar(im, shrink=0.75)
        # cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm)
        colorbar.ax.tick_params(labelsize=30)
    plt.tight_layout()
    plt.axis('off')
    

    def init():
        im.set_data(data[:, :, 0])
        return im,

    def animate(i):
        im.set_array(data[:, :, i])
        return im,

    # calling animation function of matplotlib
    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   frames=np.shape(data)[2],  # amount of frames being animated
                                   interval=milliseconds_per_frame, 
                                   blit=True)
    anim.save(filename + ".gif")   # save as gif
    #im.set_title(title)
    plt.suptitle(title, y=1)
    plt.show()


def get_meshgrid(shape):
    batch_size, x_size, y_size, t_size, _ = shape
    print(t_size)
    #batch_size, x_size, y_size, t_size = shape[0], shape[1], shape[2], shape[3]
    x_grid = torch.linspace(0, 1, x_size)
    x_grid = x_grid.reshape(1, x_size, 1, 1, 1).repeat([batch_size, 1, y_size, t_size, 1])

    y_grid = torch.linspace(0, 1, y_size)
    y_grid = y_grid.reshape(1, 1, y_size, 1, 1).repeat([batch_size, x_size, 1, t_size, 1])

    t_grid = torch.linspace(0, 1, t_size)
    t_grid = t_grid.reshape(1, 1, 1, t_size, 1).repeat([batch_size, x_size, y_size, 1, 1])

    grid = torch.concat((x_grid, y_grid, t_grid), dim=-1).float()
    return grid


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c




class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
    
    

def speedtest(function_wrapper):
    with cProfile.Profile() as pr:
        wrapper()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    filename = "speedtest_profile.prof"
    stats.dump_stats(filename=filename)
    cli.main([filename])
    

def ConvertVorticityToVelocity(w):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)

    device = w.device
    w = w.reshape(batchsize, nx, ny, -1)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    u = torch.stack([ux, uy], dim=-1)
    return u


def CalculateReynolds(vorticityFields, kinematic_viscosity=1e-3):
    if type(vorticityFields) is str:
        vorticityFields = torch.tensor(np.load(vorticityFields))
    
    velocityFields = ConvertVorticityToVelocity(vorticityFields)
    speeds = torch.linalg.vector_norm(velocityFields, dim=(-1))
    averageReynolds = speeds.mean().item() / kinematic_viscosity
    return averageReynolds


def Interpolate(y_hat, toResolution):
    fromResolutionY = y_hat.shape[-2]
    fromResolutionX = y_hat.shape[-3]
    assert fromResolutionY == fromResolutionX
    fromResolution = fromResolutionX
    
    y_hat = y_hat.permute(0, 3, 1, 2)
    scale_factor = toResolution / fromResolution
    y_hat = F.interpolate(y_hat, scale_factor=(scale_factor, scale_factor), mode='bicubic')
    y_hat = y_hat.permute(0, 2, 3, 1)
    return y_hat


def ComputeConfidenceInterval(y_hat, y, confidence=0.95):
    """
    Computes the confidence interval for a given survey of a data set.
    """
    l1_loss = torch.abs(y - y_hat) # l1_loss
    del y
    del y_hat
    nData = len(l1_loss)
    mean = l1_loss.mean()
    # se: Tensor = scipy.stats.sem(data)  # compute standard error
    # se, mean: Tensor = torch.std_mean(data, unbiased=True)  # compute standard error
    standardError = l1_loss.std(unbiased=False) / (nData ** 0.5)
    tStatistic = float(scipy.stats.t.ppf((1 + confidence) / 2., nData - 1))
    confidenceInterval = tStatistic * standardError
    return mean, confidenceInterval


def UnpackYHat(predictOut):
    y_hat, y = [], []
    for tuple_y in predictOut:
        y_hat.append(tuple_y[0])
        y.append(tuple_y[1])
    y_hat, y = torch.cat(y_hat), torch.cat(y)
    return y_hat, y


def CalculateStatistics(dataPath, loadedModel, batchSize=5, gif=False, precision='32', temporal_pred=False, temp_res=None, gpu=False):
    
    if gpu:
        acc = "gpu"
        torch.cuda.mem_get_info()
    else:
        acc = "cpu" 
    
    test_loader = npyToTorchDataLoader(dataPath, downSample=False, batchSize=batchSize, normalize=True, timeDistribution=[0.2, 0.8], distribution=[0, 0, 1], onlyTest=True, gif=gif, temporal_pred=temporal_pred, temp_res=temp_res)
    trainer = L.Trainer(accelerator=acc, devices="auto", strategy="auto", max_epochs=-1, precision=precision)#DeviceStatsMonitor(cpu_stats=True), '16-mixed'
    output = trainer.predict(loadedModel, test_loader)
    del test_loader

    if gpu:
        print(torch.cuda.mem_get_info()[0] / 1e9) 

    y_hat, y = UnpackYHat(output)
    del output

    avg_loss, conf_loss = ComputeConfidenceInterval(y_hat, y)
    print('FNO: average loss: {},  95% confidence interval: {}'.format(avg_loss.item(), conf_loss.item()))
    
    
    
    y_hat_fno = None
    if gif:
        y_hat_fno = y_hat
            
    del y_hat

    test_loader = npyToTorchDataLoader(dataPath, downSample=True, batchSize=batchSize, normalize=True, timeDistribution=[0.2, 0.8], distribution=[0, 0, 1], onlyTest=True, gif=gif, temporal_pred=temporal_pred, temp_res=temp_res)
    outputDownSampled = trainer.predict(loadedModel, test_loader)
    del test_loader

    toResolution = y.shape[-2]

    y_hat_downsampled, _ = UnpackYHat(outputDownSampled)
    del outputDownSampled
    y_hat_upsampled = Interpolate(y_hat_downsampled, toResolution)
    del y_hat_downsampled

    avg_loss, conf_loss = ComputeConfidenceInterval(y_hat_upsampled, y)
    print('Bicubic interpolation: average loss: {},  95% confidence interval: {}'.format(avg_loss.item(), conf_loss.item()))
    
    return y_hat_upsampled, y_hat_fno, y