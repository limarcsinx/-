import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from model_catdog import DogCatNet
#定义在GPU中进行训练
device = torch.device("cuda")

#图像预处理，将图片统一转换网络输入尺寸，然后转Tensor
transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])#transforms.Compose([...])是把多个预处理打包，按顺序执行.Resize((64,64))是把是所有图片强制缩放成高64，宽64的像素。ToTensor是把图片从PIL图片转成pytorch张量，同时像素值从0-255缩放到0-1.张量格式[通道数，高度，宽度]

#加载数据集和验证集
train_data = datasets.ImageFolder("D:\\AIProjects\\deep_learning\\catanddog_recognition\\Data\\big_data\\train",transform=transform)#ImageFolder是专门读取这种结构的数据，transform=transform：加载图片时自动执行上面定义的缩放，转张量操作
val_data = datasets.ImageFolder("D:\\AIProjects\\deep_learning\\catanddog_recognition\\Data\\big_data\\val",transform=transform)

#打包成DataLoader
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)#Dataloader:把数据打包成一批一批的batch给网络。shuffle=true是打乱图片顺序
val_loader = DataLoader(val_data,batch_size=64,shuffle=True)

#实例化神经网络
net = DogCatNet()
net = net.to(device)#.to(device)把整个神经网络搬到GPU上，数据必须和网络在同一个设备才能计算

#定义损失函数,优化器
loss_fn= nn.CrossEntropyLoss()#交叉熵损失函数，分类任务标配，计算网络预测结果和真实标签的差距，loss越小，预测越准
loss_fn = loss_fn.to(device)
learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)#Adam优化器，负责更新网络里面的权重参数。net.parameters()：取出网络里面所有需要学习的权重，偏置

#训练循环模块
epoch = 20
for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))
    net.train()#开启训练模式
    for imgs,targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)#imgs：一批图片张量，shape[64，3，64，64]（batch，通道，高，宽）。targets：表示这批图片对应的真实标签，0代表猫，1代表狗

        output = net(imgs)#把图片喂进网络，网络输出预测得分，shape是[64，2]64为batch_size，2为分类数量
        loss = loss_fn(output,targets)#拿预测得分和真实标签，计算损失loss

        #反向传播
        optimizer.zero_grad()#清空上一轮保存的梯度，梯度会自动累加，如果不清零，新旧梯度混在一起，更新参数就会出错
        loss.backward()#反向求导，自动计算每一个权重对loss的梯度，梯度告诉我们：这个权重往哪个方向修改，可以让loss变小，预测更准
        optimizer.step()#优化器根据梯度，自动修改网络里面所有权重w

    #每轮结束验证val
    net.eval()#评估模式
    total_loss = 0#累加验证集总的损失
    total_acc = 0#累加验证集预测正确的图片总数
    with torch.no_grad():#上下文管理器，禁止计算梯度，验证不需要反向传播，不需要更新权重，关闭梯度计算，节省大量显存，跑得更快
        for imgs,targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)
            loss = loss_fn(output,targets)
            total_loss += loss
            acc = (output.argmax(1) == targets).sum()#output.argmax(1)：对output的第一维找最大值下标，返回0或1.预测等于真实标签，为true，不等为False。.sum()统计这一批里面预测正确的数量，加到total_acc
            total_acc += acc

    print(f"val loss:{total_loss},val acc:{total_acc/len(val_data)}")

    #保存模型
    torch.save(net.state_dict(),f"D:\\AIProjects\\deep_learning\\catanddog_recognition\\dogcat_{i}.pth")
    print("模型已保存")
