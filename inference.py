import torch
from models.dlf_model import DLFModel
from data.dataset import CustomDataset
from torchvision import transforms

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DLFModel(num_classes=10).to(device)
model.load_state_dict(torch.load("dlf_model.pth", map_location=device))
model.eval()

# 推理
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = transform(Image.open("test_image.jpg").convert('RGB')).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"Predicted class: {predicted_class}")
