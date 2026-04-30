import torch

class NoiseScheduler():
    # Linear noise scheduler with beta schedule
    def __init__(self, beta1=0.0001, betaT=0.02, num_steps=1000, device='cpu'):
        self.device = device
        self.num_steps = num_steps
        self.betas = torch.linspace(beta1, betaT, num_steps).to(device).reshape(-1, 1, 1, 1)
        self.alphas = 1. - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)
        
    def forward_process(self, x0, noise, t):
        sqrt_alpha_bar = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t]

        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt
    
    def reverse_process(self, xt, noise_pred, t):
        x0 = (xt - self.sqrt_one_minus_alpha_bar[t] * noise_pred) / self.sqrt_alpha_bar[t]
        # x0 is the model's internal estimate of the original image
        x0 = torch.clamp(x0, -1., 1.)

        mean = (xt - (self.betas[t] * noise_pred) / torch.sqrt(1. - self.alpha_bar[t])) * torch.sqrt(1. / self.alphas[t])
        if t[0] == 0:
            return mean, x0
        else:
            var = (1. - self.alpha_bar[t-1]) / (1. - self.alpha_bar[t]) * self.betas[t]
            sigma = torch.sqrt(var)
            z = torch.randn_like(xt)
            sample = mean + sigma * z
            return sample, x0