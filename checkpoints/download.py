import torch
import torchvision.models as models

# Load the ViT-B/16 model with pretrained weights
# ViT_B_16_Weights.IMAGENET1K_V1 refers to the weights pretrained on ImageNet-1k
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# If you want to save the model's state dictionary to a .pth file
torch.save(model.state_dict(), 'vit_b_16_imagenet1k_v1.pth')

print("ViT-B/16 model with ImageNet-1k pretrained weights downloaded and saved as 'vit_b_16_imagenet1k_v1.pth'")
