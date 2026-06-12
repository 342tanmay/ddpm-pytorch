import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from ddpm import DDPM
from noise_scheduler import NoiseScheduler

PATH = 'Training/ddpm_celeba_epoch50.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
celeba_inference = DDPM(
    time_embedding_dim=256,
    num_attn_heads=4
).to(device)

celeba_inference.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))
celeba_inference.eval()

scheduler = NoiseScheduler(beta1=0.0001, betaT=0.02, num_steps=1000, device=device)

def generate_samples(celeba_inference, noise_scheduler, device, num_samples=25):
    celeba_inference.eval()
    print('Generating samples...')
    xt = torch.randn(num_samples, 3, 64, 64).to(device)
    with torch.no_grad():
        for t in reversed(range(noise_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = celeba_inference(xt, t_tensor)
            xt, _ = noise_scheduler.reverse_process(xt, noise_pred, t_tensor)
    x0 = (xt + 1) / 2
    x0 = torch.clamp(x0, 0.0, 1.0)
    return x0.cpu()

def plot_and_save_samples(samples):
    print('Plotting samples...')
    grid = make_grid(samples, nrow=5, padding=2, pad_value=0)
    img = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    filename = 'CelebA_Samples.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

samples = generate_samples(celeba_inference, scheduler, device, num_samples=25)
plot_and_save_samples(samples)
print('Generated samples saved to CelebA_Samples.png')