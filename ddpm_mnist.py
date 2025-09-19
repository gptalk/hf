#!/usr/bin/env python3
"""
Minimal DDPM example (PyTorch) for MNIST / small images.
Run: python ddpm_mnist.py
"""
import math
import os
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils

# -----------------------
# Utilities
# -----------------------
def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d

# sinusoidal timestep embedding (like Transformer / DDPM papers)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1,0,0))
        return emb  # shape (batch, dim)

# small residual block with time embedding
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.bn2 = nn.GroupNorm(8, out_ch)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.silu(h)
        # add time embedding
        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.conv2(h)
        h = self.bn2(h)
        h = F.silu(h)
        return h + self.shortcut(x)

# tiny U-Net
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # encoder
        self.enc1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = ResBlock(base_channels, base_channels*2, time_emb_dim)
        # bottleneck
        self.mid = ResBlock(base_channels*2, base_channels*2, time_emb_dim)
        # decoder
        self.dec2 = ResBlock(base_channels*4, base_channels, time_emb_dim)
        self.dec1 = ResBlock(base_channels*2, base_channels, time_emb_dim)
        self.out = nn.Conv2d(base_channels, in_channels, 1)

        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.downsample(e1), t_emb)
        m = self.mid(self.downsample(e2), t_emb)
        d = self.upsample(m)
        d = torch.cat([d, e2], dim=1)
        d = self.dec2(d, t_emb)
        d = self.upsample(d)
        d = torch.cat([d, e1], dim=1)
        d = self.dec1(d, t_emb)
        out = self.out(d)
        return out

# -----------------------
# DDPM noise schedule and helpers
# -----------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        betas = linear_beta_schedule(timesteps).to(device)  # (T,)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        # useful values for sampling
        # posterior variance (with safe padding for t=0)
        self.posterior_variance = torch.cat([
            torch.tensor([1e-8], device=device),
            betas[1:] * (1. - alphas_cumprod[:-1]) / (1. - alphas_cumprod[1:])
        ])
        # pad to length T for indexing convenience
        self.posterior_variance = torch.cat([self.posterior_variance, torch.tensor([1e-8], device=device)])

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the clean image x_start to timestep t (q(x_t | x_0))
        x_start: (B,C,H,W)
        t: (B,) long tensor of timesteps (0..T-1)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_sample(self, model, x_t, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)

        # 预测噪声
        eps_theta = model(x_t, t.float())

        # 根据公式还原出 x0
        x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * eps_theta) / torch.sqrt(alphas_cumprod_t)

        # 均值项
        mean = (1 / torch.sqrt(alphas_t)) * (
            x_t - (1 - alphas_t) / sqrt_one_minus_alphas_cumprod_t * eps_theta
        )

        # 方差项
        if t[0] > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        return mean + torch.sqrt(betas_t) * noise

    # @torch.no_grad()
    # def sample(self, model, shape, device):
        # model.eval()
        # x = torch.randn(shape, device=device)
        # T = self.timesteps
        # for i in reversed(range(T)):
            # t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            # x = self.p_sample(model, x, t)
        # return x

    @torch.no_grad()
    def sample(self, model, shape, device, save_intermediate=False, save_dir="denoise_steps", steps_to_save=10):
        model.eval()
        x = torch.randn(shape, device=device)
        T = self.timesteps
        os.makedirs(save_dir, exist_ok=True)

        # 计算要保存的时间点（等间隔）
        if save_intermediate:
            save_ts = list(reversed(sorted({int(i * (T/steps_to_save)) for i in range(steps_to_save)})))
        else:
            save_ts = []

        imgs_to_save = []
        for i in reversed(range(T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
            if save_intermediate and i in save_ts:
                imgs_to_save.append((i, x.clone().cpu()))

        # 保存图像序列（每个时间点一张 grid）
        if save_intermediate:
            from torchvision import utils
            for idx, (ti, img_tensor) in enumerate(imgs_to_save):
                img = (img_tensor.clamp(-1,1) + 1) / 2
                utils.save_image(img, os.path.join(save_dir, f"step_{ti:04d}.png"), nrow=8)
        return x


# -----------------------
# Training
# -----------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset (MNIST). normalize to [-1,1]
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # (x - 0.5)/0.5 -> [-1,1]
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = SimpleUNet(in_channels=1, base_channels=64, time_emb_dim=128).to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)  # range [-1,1]
            b = images.size(0)
            t = torch.randint(0, args.timesteps, (b,), device=device).long()  # random timesteps
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(images, t, noise=noise)
            # model predicts noise
            pred = model(x_t, t.float())  # returns same shape
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % args.log_interval == 0:
                print(f"Epoch {epoch} Step {global_step} Loss {loss.item():.6f}")

            if global_step % args.sample_interval == 0:
                with torch.no_grad():
                    samples = diffusion.sample(model, (64, 1, args.image_size, args.image_size), device=device)
                    # unnormalize from [-1,1] to [0,1]
                    samples = (samples.clamp(-1,1) + 1) / 2
                    utils.save_image(samples, os.path.join(args.output_dir, f"sample_{global_step}.png"), nrow=8)
                    # save model
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_latest.pth"))
            global_step += 1

    # final save
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pth"))
    print("Training finished.")

# -----------------------
# Main and args
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=200)  # smaller T for quick demo
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--output_dir", type=str, default="outputs_ddpm")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--sample_interval", type=int, default=500)
    args = parser.parse_args()
    train(args)
