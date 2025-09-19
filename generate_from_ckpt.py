# generate_from_ckpt.py
import torch
from torchvision import utils
from ddpm_mnist import SimpleUNet, Diffusion  # 假设你的主文件名和类名相同
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = "outputs_ddpm/model_final.pth"  # 改成你的路径
os.makedirs("samples_from_ckpt", exist_ok=True)

# 加载模型（确保参数一致）
model = SimpleUNet(in_channels=1, base_channels=64, time_emb_dim=128).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))

diff = Diffusion(timesteps=200, device=device)  # timesteps 要和训练时一致

model.eval()
with torch.no_grad():
    # samples = diff.sample(model, (64, 1, 28, 28), device=device)  # 64张图
    samples = diff.sample(model, (64,1,28,28), device=device, save_intermediate=True, save_dir="denoise_steps", steps_to_save=12)
    samples = (samples.clamp(-1,1) + 1) / 2  # [-1,1] -> [0,1]
    utils.save_image(samples, "samples_from_ckpt/grid.png", nrow=8)
print("Saved samples_from_ckpt/grid.png")
