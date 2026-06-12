import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from noise_scheduler import NoiseScheduler
from dataloaders import get_dataloaders
from ddpm import DDPM

def generate_samples(model, noise_scheduler, device, num_samples=16):
    model.eval()
    xt = torch.randn(num_samples, 3, 64, 64).to(device)
    with torch.no_grad():
        for t in reversed(range(noise_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(xt, t_tensor)
            xt, _ = noise_scheduler.reverse_process(xt, noise_pred, t_tensor)
    x0 = (xt + 1) / 2
    x0 = torch.clamp(x0, 0.0, 1.0)
    return x0.cpu()

def plot_samples(samples, epoch=None):
    num_samples = samples.shape[0]
    grid_size = int(num_samples ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = samples[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    filename = f'CelebA_Samples@Epoch{epoch}.png' if epoch is not None else 'CelebA_Samples.png'
    plt.savefig(filename)
    plt.close()

def train(model, noise_scheduler, train_loader, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=epochs * len(train_loader),
        pct_start=0.1
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for images, _ in pbar:
            x0 = images.to(device)
            B = images.size(0)
            t = torch.randint(0, noise_scheduler.num_steps, (B,), device=device)
            noise = torch.randn_like(x0)

            xt = noise_scheduler.forward_process(x0, noise, t)
            noise_pred = model(xt, t)

            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Avg. Loss: {avg_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}')

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'ddpm_celeba_epoch{epoch+1}.pt')
            samples = generate_samples(
                model=model,
                noise_scheduler=noise_scheduler,
                device=device,
            )
            plot_samples(samples, epoch=epoch+1)
        
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    noise_scheduler = NoiseScheduler(
        beta1=0.0001,
        betaT=0.02,
        num_steps=1000,
        device=device
    )
    
    model = DDPM(
        time_embedding_dim=256,
        num_attn_heads=4
    ).to(device)

    celeba_train_loader, _ = get_dataloaders(batch_size=64)

    print('Starting training...')
    celeba_ddpm = train(
        model=model,
        noise_scheduler=noise_scheduler,
        train_loader=celeba_train_loader,
        epochs=50,
        lr=1e-4,
        device=device
    )