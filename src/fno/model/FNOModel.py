import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from MLP import *
from FourierLayer import *
from Utilities import *
# from LossFunction import *


class ModelCallbacks(L.Callback):
    def on_train_start(self, trainer):
        # RichModelSummary(max_depth=1)
        pass
        
    def on_train_end(self, trainer):
        pass


class FNOModel(L.LightningModule):
    def __init__(self, in_neurons, hidden_neurons, out_neurons, modesSpace, modesTime, time_padding, input_size, learning_rate, restart_at_epoch_n, train_loader, loss_function):
        super().__init__()
        #self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.restart_at_epoch_n = restart_at_epoch_n
        self.padding = time_padding # set padding here based on input_size
        self.n_batches = len(train_loader)
        self.n_training_samples = len(train_loader.dataset)
        self.loss_name = loss_function
        #train_batch, _ = next(iter(train_loader))
        #x_shape = train_batch.size()
        #self.register_buffer("meshgrid", get_meshgrid(x_shape))
        
        # Network architechture
        self.p = nn.Linear(input_size, out_neurons)
        
        self.fourier1 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp1 = MLP(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w1 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.fourier2 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp2 = MLP(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w2 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.fourier3 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp3 = MLP(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w3 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.fourier4 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp4 = MLP(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w4 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.q = MLP(in_neurons, 4 * hidden_neurons, 1, kernel_size=1) # Single output predicts T timesteps
        
        if loss_function == 'L2':
            self.loss_function = LpLoss()
        elif loss_function == 'MSE':
            self.loss_function = F.mse_loss
        elif loss_function == 'MAE':
            self.loss_function = F.l1_loss
    
            
    def forward(self, x): # input dim: [B, X, Y, T, T_in]
        meshgrid = get_meshgrid(x.shape).to(self.device)
        x = torch.concat((x, meshgrid), dim=-1) # [B, X, Y, T, 3 + T_in]
        del meshgrid
        x = self.p(x) # [B, X, Y, T, H]
        x = x.permute(0, 4, 1, 2, 3) # [B, H, X, Y, T]
        x = F.pad(x, [0, self.padding]) # Zero-pad
        x1 = self.fourier1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        del x1
        del x2
        x = F.gelu(x)
        
        x1 = self.fourier2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        del x1
        del x2
        x = F.gelu(x)
        
        x1 = self.fourier3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        del x1
        del x2
        x = F.gelu(x)
        
        x1 = self.fourier4(x)
        x1 = self.mlp4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        del x1
        del x2
        
        x = x[..., :-self.padding] # Unpad zeros
        x = self.q(x) # [B, 1, X, Y, T]
        x = x.permute(0, 2, 3, 4, 1)  # [B, X, Y, T, 1]
        x = x.squeeze_(dim=-1)
        return x
    

    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x) # [B, X, Y, T]
        train_loss = self.loss_function(y_hat, y) # .view(len(y), -1)
        train_mse = F.mse_loss(y_hat, y)
        log_dict = {'mse_loss': train_mse, 'train_' + self.loss_name + '_loss': train_loss}
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)
        return train_loss 


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # [B,X,Y,T]
        val_loss = self.loss_function(y_hat, y)
        self.log('val_' + self.loss_name + '_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # [B,X,Y,T]
        
        test_loss = self.loss_function(y_hat, y)

        self.log('test_' + self.loss_name + '_loss', test_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return test_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        del batch
        return self(x), y
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.restart_at_epoch_n)
        return optimizer