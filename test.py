import torch
from PIL import Image#用来打开本地图片文件
import torchvision.transforms as transforms
from model_catdog import DogCatNet

# 自动选择GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 和训练保持一致的预处理
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 打开图片，换成你自己图片路径
img = Image.open("test\\0cc45963b3fdb8e712e50324e41c541d.png").convert("RGB")#.convert("RGB"):把图片强制转化为RGB三通道，防止通道数不对
img = test_transform(img)
img = torch.unsqueeze(img, 0)#增加一个batch维度batch_size=1
img = img.to(device)

# 加载模型权重
net = DogCatNet().to(device)#先创建一个空网络，放到对应设备
net.load_state_dict(torch.load("dogcat_19.pth"))#加载训练好的权重文件，因为pth文件只是权重，不包含网络结构，所以必须先定义DogCatNet
net.eval()

with torch.no_grad():
    res = net(img)
    pred = res.argmax(1).item()

if pred == 0:
    print("预测：猫")
else:
    print("预测：狗")
