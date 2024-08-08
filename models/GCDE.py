import torch
import torch.nn.functional as F
import torch.nn as nn
import controldiffeq
from models.vector_fields import *

class NeuralGCDE(nn.Module):
    def __init__(self, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol, rtol, solver, num_nodes, horizon, num_layers, embed_dim):
        super(NeuralGCDE, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = horizon
        self.num_layers = num_layers

        self.default_graph = False
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)
        
        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        self.init_type = 'fc'
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
            self.start_conv_z = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))

    def forward(self, times, coeffs):
        #source: B, T_1, N, D
        #target: B, T_2, N, D

        spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        if self.init_type == 'fc':
            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()

        z_t = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative, #dh_dt
                                           h0=h0,
                                           z0=z0,
                                           func_f=self.func_f,
                                           func_g=self.func_g,
                                           t=times,
                                           method=self.solver,
                                           atol=self.atol,
                                           rtol=self.rtol) # [w,b,N,h]

        return z_t.permute(1,2,0,3) # [b,N,w,h]