import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from .actor_critic import get_activation

class MLP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim=[256, 128], 
                 activation="relu",
                 name=None,
                 output_bound=None,
                 dropout_rate=None,
                 **kwargs):
        if kwargs:
            print("MLP.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(MLP, self).__init__()
        
        activation = get_activation(activation)
        self.output_bound = output_bound
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        mlp_layers = []
        mlp_layers.append(nn.Linear(input_dim, hidden_dim[0]))
        mlp_layers.append(activation)
        if dropout_rate is not None:
            mlp_layers.append(nn.Dropout(dropout_rate))
        for i in range(len(hidden_dim)):
            if i == len(hidden_dim) - 1:
                mlp_layers.append(nn.Linear(hidden_dim[i], output_dim))
                if output_bound is not None:
                    mlp_layers.append(nn.Tanh()) # bound the output
            else:
                mlp_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                mlp_layers.append(activation)
                if dropout_rate is not None:
                    mlp_layers.append(nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(*mlp_layers)
        
        self.init_weights()
        
        print(f"MLP: {name}")
    
    def forward(self, x):
        if self.output_bound is None:
            return self.mlp(x)
        else:
            return self.output_bound * self.mlp(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def loss(self, y_hat, y):
        return torch.mean(F.mse_loss(y_hat, y))
    
    def traning_step(self, features_batch, labels_batch):
        l = self.loss(self(features_batch), labels_batch)
        return l
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l