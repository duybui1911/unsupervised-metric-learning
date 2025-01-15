import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.dinov2.model import vit_small as dinov2_small


model = dinov2_small(patch_size=14, block_chunks=0, init_values=1.0e-05, img_size=518)
print(model)
ckpt_path = "/home/anlab/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
state_dict = torch.load(ckpt_path, map_location="cpu",)
msg = model.load_state_dict(state_dict)
print(msg)
patch_size = 14
image_size = 518
image = Image.open("/media/anlab/DATA2/ruybk/dino/attns/img.png").convert("RGB")
output_dir = "attns"

transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

image = transform(image).unsqueeze(0)
model.get_last_selfattention(image)
# print(msg)
w_featmap = image.shape[-2] // patch_size
h_featmap = image.shape[-1] // patch_size

attentions = np.load("attn.npy")
attentions = torch.from_numpy(attentions)
nh = attentions.shape[1]
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
attentions = attentions.reshape(nh, w_featmap, h_featmap)
attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="bicubic")[0].cpu().numpy()

os.makedirs(output_dir, exist_ok=True)
# torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
for j in range(nh):
    fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
    plt.imsave(fname=fname, arr=attentions[j], format='png')
    print(f"{fname} saved.")