import torch
from torch.utils.data import DataLoader
from models.dlf_model import DLFModel
from data.dataset import CustomDataset
from torch.cuda.amp import GradScaler, autocast
import yaml
from utils.logger import setup_logger

# 加载配置
with open("configs/base.yaml", "r") as f:
    config = yaml.safe_load(f)

# 初始化
logger = setup_logger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DLFModel(num_classes=config['data']['num_classes']).to(device)
dataset = CustomDataset(data_dir=config['data']['data_dir'], image_size=config['data']['image_size'])
dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
scaler = GradScaler(enabled=config['training']['mixed_precision'])

# 训练循环
for epoch in range(config['training']['num_epochs']):
    model.train()
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=config['training']['mixed_precision']):
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

        scaler.scale(loss).backward()
        if (i + 1) % config['training']['gradient_accumulation'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        logger.info(f"Epoch [{epoch+1}/{config['training']['num_epochs']}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}")

# 保存模型
torch.save(model.state_dict(), "dlf_model.pth")
