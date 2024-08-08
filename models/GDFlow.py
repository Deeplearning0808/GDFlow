import torch
import torch.nn as nn

from models.NF import MAF
from models.vector_fields import *
from models.GCDE import *

class GDFlow(nn.Module):
    def __init__ (self, device, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, entity_mean, entity_covar, dropout = 0.1, batch_norm=True, q=0.05, ab_ncde=False, ab_qunatile=False):
        super(GDFlow, self).__init__()

        self.vector_field_f = FinalTanh_f(input_channels=input_size,
                                          hidden_channels=hidden_size,
                                          hidden_hidden_channels=hidden_size,
                                          num_hidden_layers=1)
        self.vector_field_g = VectorField_g(input_channels=input_size,
                                            hidden_channels=hidden_size,
                                            hidden_hidden_channels=hidden_size,
                                            num_hidden_layers=1,
                                            num_nodes=n_sensor,
                                            cheb_k=2,
                                            embed_dim=10,
                                            g_type='agc')
        self.ncde = NeuralGCDE(func_f=self.vector_field_f,
                               func_g=self.vector_field_g,
                               input_channels=input_size,
                               hidden_channels=hidden_size,
                               output_channels=input_size,
                               initial=True,
                               device=device,
                               atol=1e-1,
                               rtol=1e-3,
                               solver='rk4',
                               num_nodes = n_sensor,
                               horizon = 1,
                               num_layers = 1,
                               embed_dim=10)
        self.times = self.times = torch.linspace(0, window_size-1, window_size).to(device)

    
        self.nf = MAF(n_blocks=n_blocks,
                        n_sensor=n_sensor,
                        window_size=window_size,
                        input_dim=input_size,
                        Lin_h=4,
                        n_hidden=n_hidden,
                        input_order='sequential',
                        cond_label_size=hidden_size,
                        activation='tanh',
                        batch_norm=batch_norm,
                        mode='rand',
                        device=device,
                        entity_mean=entity_mean,
                        entity_covar=entity_covar)
            
        self.q = q # Qunatile value
        self.calc_loss = {
            0.0: self.min_loss,
            1.0: self.mean_loss
        }.get(self.q, self.quantile_loss)
        
        self.ab_ncde = ab_ncde
        self.ab_qunatile = ab_qunatile
        print('ab_ncde:', self.ab_ncde)
        print('ab_qunatile:', self.ab_qunatile)
        if self.ab_ncde:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        
    def min_loss(self, ll): # If the loss is 0.0 ==> Calculate loss using a minimum value of log-likelihood
        return torch.min(ll)

    def mean_loss(self, ll): # If the loss is 1.0 ==> Calculate loss using a mean value of log-likelihood (General mean loss approach)
        return torch.mean(ll)

    def quantile_loss(self, ll): # If loss is neither 0.0 nor 1 ==> Calculate loss using a qunatile value (Maximize likelihood of low likelihood inlier samples located in the distribution boundary)
        return torch.quantile(ll, self.q)
        
    def forward(self, x, coeffs): # [b,N,w,1]
        # log_prob = self.test(x, coeffs).mean() # Scalar value
        log_prob = self.test(x, coeffs)
        if 0 <= self.q <= 1 and not self.ab_qunatile: # If q is between 0 and 1, apply the Quantile function
            qll = self.calc_loss(log_prob)
            return qll
        else:
            return log_prob.mean()

    def test(self, x, coeffs): # [b,N,w,1]
        x_ori = x

        if self.ab_ncde:
            full_shape = x.shape
            x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
            h, _ = self.rnn(x)  # [b,N,w,h]
            h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        else:
            h = self.ncde(self.times, coeffs) # [b,N,w,h]
        # Normalizing flow
        log_prob = self.nf.log_prob(x_ori, h) # [b]
        return log_prob