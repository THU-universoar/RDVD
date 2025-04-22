# 导入系统库
import time

# 导入PyTorch库
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# 导入自定义库
from datasets import LSTMDataset
from models import LSTMModel


def main():
    # 测试集目录
    data_folder = "./data"

    # 定义设备运行环境
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # 加载预训练模型
    checkpoint = torch.load("./results/checkpoint.pth", map_location=device)
    model = LSTMModel(input_size=6)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # 定制化的dataloader
    val_dataset = LSTMDataset(data_folder, mode="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # 定义评估指标
    criterion = nn.MSELoss().to(device)

    # 记录误差值
    MSEs = []

    # 记录测试时间
    model.eval()
    start = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            MSEs.append(loss.item())

    avg_mse = sum(MSEs) / len(MSEs)
    print(f'MSE: {avg_mse:.3f}')
    print(f'平均单张样本用时: {(time.time() - start) / len(val_dataset):.3f} 秒')


if __name__ == '__main__':
    main()
