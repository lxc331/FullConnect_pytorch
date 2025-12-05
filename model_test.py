import time
from token import MINUS

from pandas import read_excel
from torchvision.datasets import FashionMNIST # 从 torchvision.datasets 模块中导入 FashionMNIST 数据集
from torchvision import transforms # 导入 torchvision.transforms 模块，用于图像变换
import numpy as np # 导入 numpy 模块，用于数值计算
import torch.utils.data as data # 导入 torch.utils.data 模块，用于处理数据集
import matplotlib.pyplot as plt # 导入 matplotlib.pyplot 模块，用于可视化

from fully_connect_network import FullConnect # 从 model.py 中导入 AlexNet 模型
# 从 model.py 中导入 FullConnect 模型
import torch # 导入 torch 模块，用于张量计算
from torch import nn # 导入 torch.nn 模块，用于定义神经网络层
import copy # 导入 copy 模块，用于复制对象
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler # 导入sklearn库中的线性回归模型


# 解决中文显示问题，以及符号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义一个函数，用于处理训练集和验证集的数据
def deal_train_and_val_data():
    # 数据预处理
    dataset = read_excel('./data/乳腺癌原始数据.xlsx')

    # 提取特征
    x = dataset.iloc[:, 1:-1]  # 提取所有行，除了最后一列
    # print(f"数据特征维度: {x.shape}")  # 添加这行来查看实际特征维度

    # 提取标签
    y = dataset.iloc[:, -1]
    # print(y)

    # 随机划分训练集和测试集, random_split 函数的作用是将数据集随机划分成训练集和测试集，这里是将数据集随机划分成 80% 训练集和 20% 测试集
    train_data_x, test_data_x = data.random_split(x, [round(len(dataset) * 0.8), round(len(dataset) * 0.2)],
                                                 generator=torch.Generator().manual_seed(42))
    train_data_y, test_data_y = data.random_split(y, [round(len(dataset) * 0.8), round(len(dataset) * 0.2)],
                                                 generator=torch.Generator().manual_seed(42))

    # 从Subset对象中提取实际的标签数据并转换为Tensor
    # torch.tensor 函数的参数说明：
    # y.iloc[test_data_y.indices].values：从 test_data_y 中提取实际的标签数据，并将其转换为 numpy 数组
    # dtype=torch.long：将 numpy 数组转换为 torch 张量，数据类型为 long（整数类型）
    test_labels = torch.tensor(y.iloc[test_data_y.indices].values, dtype=torch.long)

    # 从Subset对象中提取实际的特征数据并转换为Tensor
    # torch.tensor 函数的参数说明：
    # x.iloc[test_data_x.indices].values：从 test_data_x 中提取实际的特征数据，并将其转换为 numpy 数组
    # dtype=torch.float32：将 numpy 数组转换为 torch 张量，数据类型为 float32（单精度浮点数类型）
    test_data_x_tensor = torch.tensor(x.iloc[test_data_x.indices].values, dtype=torch.float32)

    # 对测试集特征数据进行归一化处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test_data_x = scaler.fit_transform(test_data_x_tensor)  # 注意：测试集使用训练集的参数进行归一化

    # 将归一化后的特征数据转换为 torch 张量
    test_data_x = torch.tensor(scaled_test_data_x, dtype=torch.float32)
    test_data_y = test_labels

    # 定义测试集的数据集
    test_dataset = data.TensorDataset(test_data_x, test_data_y)

    # 定义测试集的 DataLoader
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_dataloader

# 定义一个函数，用于测试模型
def test_model(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到 指定的设备上
    model = model.to(device)
    # 初始化测试集的准确率为 0
    test_accuracy = 0.0
    # 初始化测试集的样本数为 0
    test_num = 0
    # 只进行前向传播，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将测试集的样本移动到 指定的设备上
            test_data_x = test_data_x.to(device)
            # 将测试集的标签移动到 指定的设备上
            test_data_y = test_data_y.to(device)
            # 切换模型到评估模式, 评估模式下，模型的行为会和训练模式下不同, 例如 Dropout 层会被关闭, BatchNorm 层会使用训练时的统计信息
            # 这是因为在训练模式下，Dropout 层会随机将输入的一些元素设为 0，而在评估模式下，Dropout 层会将所有元素都设为 1，
            # 这是为了保持模型的稳定性，避免在评估时因为随机失活而导致的结果不一致
            model.eval()
            # 前向传播，计算模型的输出
            test_output = model(test_data_x)
            # 计算模型的预测结果，这里是取输出中概率最大的那个类作为预测结果
            # dim=1 表示在每一行（10个类别）中查找概率最大的索引, 作为模型的预测标签
            # argmax 函数返回的是概率最大的索引，这里是将索引转换为类别标签
            test_predict = torch.argmax(test_output, dim=1)
            # 计算测试集的准确率，这里是判断预测结果是否等于真实标签，如果相等，就将准确率加 1
            # 这里的 sum() 函数是将所有相等的元素的和相加，这里是将所有预测正确的元素的和相加，得到预测正确的样本数
            # 这里的 test_predict == test_data_y 是一个布尔张量，这里是将所有预测正确的元素设为 True，将所有预测错误的元素设为 False
            test_accuracy += torch.sum(test_predict == test_data_y)
            # 测试集的样本数加 1
            test_num += test_data_x.size(0)

    # 计算测试集的准确率，这里是将准确率除以样本数，得到准确率
    # 这里的 item() 函数是将张量转换为 Python 中的标量，这里是将预测正确的样本数转换为 Python 中的标量
    test_accuracy = test_accuracy.double().item() / test_num
    print(f"测试集的准确率为: {test_accuracy:.4f}")

# 定义一个函数，用于测试每个 batch(这里是 1 个样本) 上的预测与真实标签是否相等
def test_model_on_batch(model, test_dataloader):
    # 将模型移动到 指定的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义类别名称, 这里是 FashionMNIST 数据集的类别名称
    classes = ['良性肿瘤', '恶性肿瘤']
    # 只进行前向传播，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad(): # no_grad 上下文管理器, 用于在不计算梯度的情况下进行前向传播, 从而节省内存, 加快运行速度
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 开启评估模式, 评估模式下，模型的行为会和训练模式下不同, 例如 Dropout 层会被关闭, BatchNorm 层会使用训练时的统计信息
            model.eval()
            # 前向传播，计算模型的输出
            output = model(b_x)
            # 计算模型的预测结果，这里是取输出中概率最大的那个类作为预测结果
            # dim=1 表示在每一行（10个类别）中查找概率最大的索引, 作为模型的预测标签
            # argmax 函数返回的是概率最大的索引，这里是将索引转换为类别标签
            pre_label = torch.argmax(output, dim=1)
            # 也可以将模型的预测结果转换为 numpy 数组(result_label = pre_label.numpy()), 将模型的预测结果从 torch 张量转换为 numpy 数组
            # 结果是还原成了numpy数组[]
            # 这里是将模型的预测结果从 torch 张量转换为 Python 中的标量(单个元素), 方便后续的分析和可视化
            result_label = pre_label.item()
            # 这里是将模型的真实标签从 torch 张量转换为 Python 中的标量(单个元素个, 方便后续的分析和可视化
            label = b_y.item()
            print(f'预测值: {classes[result_label]},-------,  真实值: {classes[label]}')

if __name__ == '__main__':
    # 加载测试集的 DataLoader
    test_data_loader = deal_train_and_val_data()
    #加载模型
    model = FullConnect()
    # 加载模型参数
    model.load_state_dict(torch.load('./model/best_model.pth'))
    # 计算模型在测试集上的准确率
    test_model(model, test_data_loader)
    # 测试模型在测试集上的性能
    test_model_on_batch(model, test_data_loader)



