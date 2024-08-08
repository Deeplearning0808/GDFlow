import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math


def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    '''
        This function generates autoregressive masks.
        In a mask, a value of 1 allows a connection, and a mask with a value of 0 blocks it.
    '''
     
    degrees = [] # degrees of connections between layers -- ensure at most in_degree - 1 connections

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_size > 1:
        if input_order == 'sequential':
            degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                degrees += [torch.arange(hidden_size) % (input_size - 1)]
            degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

        elif input_order == 'random':
            degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
                degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [input_degrees - 1]
    
    else: # In this case, it's fully-connected
        degrees += [torch.zeros([1]).long()]
        for _ in range(n_hidden+1):
            degrees += [torch.zeros([hidden_size]).long()]
        degrees += [torch.zeros([input_size]).long()]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """
        It is a custom linear layer that applies a mask to the weights during the forward pass,
        effectively implementing the autoregressive property by zeroing out weights that would violate the order of dependences specified by the mask. 
    """
    def __init__(self, input_size, output_size, mask, cond_label_size=None):
        super().__init__(input_size, output_size)
        '''
            In many modeling scenarios, especially in generative models, there's a need to generate data not just in an unrestricted manner
            but under certain conditions or contexts. This conditional data (y) represents additional information that dictates or influences the generation process.
            For example, in image generation, y could represent class labels (e.g., generating images of cats vs. dogs),
            or in speech synthesis, it could represent speaker characteristics.
        '''
        self.register_buffer('mask', mask)
        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(output_size, input_size*cond_label_size) / math.sqrt(input_size*cond_label_size))

    def forward(self, x, y=None): # x=[b,N,w,1], y=[b,N,w,h] |
        if x.shape[-1] == 1 and x.dim() == 4: # Handling an exception for the case when the last batch of a for-loop happens to be 1, resulting in a 2D array due to squeezing
            x = x.squeeze(-1)  # Remove the last dimension if it is 1
            
        b, N, w = x.shape
        x_reshaped = x.view(-1, w)

        out = F.linear(x_reshaped, self.weight * self.mask, self.bias).reshape(b,N,-1) # [b,N,w] -> [b,N,Lin_h] | 

        # Conditional Flow
        if y is not None:
            b, N, w, h = y.shape
            y_reshaped = y.reshape(b*N, w*h)
            cond_mask = self.mask.repeat(1,self.cond_label_size)
            out = out + F.linear(y_reshaped, self.cond_weight * cond_mask).reshape(b,N,-1) # [b,N,Lin_h] | 
        
        return out


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, n_sensor, window_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(n_sensor, window_size))
        self.beta = nn.Parameter(torch.zeros(n_sensor, window_size))

        self.register_buffer('running_mean', torch.zeros(n_sensor, window_size))
        self.register_buffer('running_var', torch.ones(n_sensor, window_size))

    def forward(self, x, cond_y=None): # x [b,N,w], cond_y [b,N,w,h]
        if self.training:
            self.batch_mean = x.mean(0) # [N,w]
            self.batch_var = x.var(0) # [N,w] note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum)) # [N,w]
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum)) # [N,w]

            mean = self.batch_mean
            var = self.batch_var
        
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps) # [b,N,w]
        y = self.log_gamma.exp() * x_hat + self.beta # [b,N,w]

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps) # [b,N,w]

        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians


''' Masked Autoencoder for Distribution Estimation '''
class MADE(nn.Module):
    def __init__(self, window_size, Lin_h, n_hidden, cond_label_size, activation, input_order, input_degrees=None):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(window_size)) # mean=0
        self.register_buffer('base_dist_var', torch.ones(window_size)) # var=1

        # create masks
        masks, self.input_degrees = create_masks(input_size=window_size, hidden_size=Lin_h, n_hidden=n_hidden, input_order='sequential', input_degrees=None)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(masks[0].shape[-1], masks[0].shape[0], masks[0], cond_label_size) # (1)ML{w->Lin_h}
        self.net = []
        for m in masks[1:-1]: # masks[1]
            self.net += [activation_fn, MaskedLinear(m.shape[-1], m.shape[-1], m)] # (2)tanh->ML{Lin_h->Lin_h}
        self.net += [activation_fn, MaskedLinear(masks[-1].shape[-1], 2 * masks[-1].shape[0], masks[-1].repeat(2,1))] # (4)tanh->ML{Lin_h->2w}
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None): # x=[b,N,w,1], y=[b,N,w,h]
        m, log_var = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1) # mean=[b,N,w], log variance=[b,N,w]
        '''
            - m: Represents the predicted mean of the transformed variable z for a given input x
            - log_var: It is used to scacle the input, cocntrolling how much each dimensions of the input contributes to the output.
              The exponential of log_var (standard deviation) is a scaling factor applied to the difference x-m
        '''
        z = (x.squeeze() - m) * torch.exp(-log_var) # [b,N,w], log_var adjustment applies a non-uniform stretch or compression to the data, based on the predicted variance.
        # Using torch.exp(-log_var) for scaling results in:
        log_abs_det_jacobian = - log_var # [b,N,w]
        '''
            In MADE, when the transformation z~ is applied, the partial derivatives (hence the elements of the Jacobian matrix) along the diagonal are exp(-log_var_i),
            because each u_i depends only on x_i.
        '''
        return z, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        z, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(z) + log_abs_det_jacobian, dim=1)


''' Masked Autoregressive Flow '''
class MAF(nn.Module): 
    def __init__(self, n_blocks, n_sensor, window_size, input_dim, Lin_h, n_hidden, input_order, cond_label_size, activation, batch_norm, mode, device, entity_mean, entity_covar):
        super().__init__()

        if mode == 'zero':
            self.base_dist_mean = torch.randn(n_sensor).repeat_interleave(window_size).to(device) # [Nw]
            self.base_dist_covar = torch.eye(n_sensor*window_size).to(device) # [Nw]
            self.base_distribution = D.MultivariateNormal(self.base_dist_mean, self.base_dist_covar)
        elif mode == 'rand':
            self.base_dist_mean = torch.randn(n_sensor).repeat_interleave(window_size).to(device) # [Nw]
            self.base_dist_covar = torch.eye(n_sensor*window_size).to(device) #  [Nw]
            self.base_distribution = D.MultivariateNormal(self.base_dist_mean, self.base_dist_covar)
        elif mode == 'predefined':
            self.base_dist_mean = torch.tensor(entity_mean).repeat_interleave(window_size).to(device) # [N] -> [Nw]
            self.base_dist_covar = torch.eye(n_sensor*window_size).to(device) # [Nw]
            self.base_distribution = D.MultivariateNormal(self.base_dist_mean, self.base_dist_covar)
        else:
            raise AttributeError('no choice')
        
        # construct model
        '''
            - MADE is a type of autoregressive neural network used within each block of MAF to ensure that the transformation respects the autoregressive property.
            - BatchNorm is used for stable training and efficient data handling within the network.
        '''
        modules = []
        self.input_size = input_dim # 1
        self.input_degrees = None
        for i in range(n_blocks): # 1
            modules += [MADE(window_size, Lin_h, n_hidden, cond_label_size, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0) # flip(0): reverses the order of the elements in the tensor along the specified dimension (0)
            modules += batch_norm * [BatchNorm(n_sensor, window_size)]
        self.net = FlowSequential(*modules)
        '''
            <Reason for flipping the input_degress for the subsequent layer> - just some stabilizing technique
            1. Alternating dependency orders
            2. Improving flow coverage
            3. Symmetry and balance in learning
        '''

    def base_dist(self, z): # z=[b,N,w]
        '''
            Calculate the log probability of the transformed data z (assumed to be sampled or ideally conform to a Gaussian distribution)
            under the Gaussian base distribution
        '''
        b, N, w = z.shape
        logp = self.base_distribution.log_prob(z.view(b,-1)) # [b]
        return logp # log-likelihood of the transformed data to match Gaussian distribution

    def forward(self, x, y=None): # x=[b,N,w,1], y=[b,N,w,h]
        ''' Forward transformation '''
        return self.net(x, y)

    def log_prob(self, x, y=None): # x=[b,N,w,1], k=N, window_size=w, y=[b,N,w,h]
        # (1) Transform input data x to z
        z, sum_log_abs_det_jacobians = self.forward(x, y)
        b, N, w = z.shape # (b, n, w)
        # z=[b,N,w] -> The transformed version of the input data x
        # sum_log_abs_det_jacobians=[b,N,w]
        '''
            Typically involves multiple sequential transformations or blocks, where each block contributes to the overall transformation from x to u.
            Each block has its own Jacobian determinant, and the sum_log_abs_det_jacobians represents the sum of the logarithms of these determinants across all the blocks.
            This summation is necessary because the overall transformation's volume change is the product of volume changes from each block
            (and logarithms turn products into sums).
        '''
        # (2) Compute the log-likelihood by applying base_dist method
        '''
            - The determinant of the Jacobian matrix quantifies the factor by which the transformation scales volumes around each point in the space where x resides.
            - The sum_log_abs_det_jacobians is summed into the computation of the log probability to properly account for how these volume changes affect the density of x.
              This ensures that the model's output is a correct reflection of the underlying probabilities after transformations.
            - This corrects for the volume change induced by the transformation, ensuring that the log probability computed is the actual log probability of x under the transformed model.
            - The sum_log_abs_det_jacobians acts something like an adjustment.
        '''
        out_log_prob = self.base_dist(z) + sum_log_abs_det_jacobians.view(b,-1).sum(-1) # [b]
        return out_log_prob