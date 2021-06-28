from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
net_type = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
method_type = ['TENSILE', 'vDNN', 'Capuchin']
with open('./log/MultiWorkloadsMSR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
x = ['x1', 'x2', 'x3']
for i in range(5):
    TENSILE = df[i, :]
    vDNN = df[i + 5, :]
    Capuchin = df[i + 10, :]
    plt.plot(x, TENSILE)
    plt.plot(x, vDNN)
    plt.plot(x, Capuchin)
    plt.legend(method_type)
    plt.xlabel('workload number')
    plt.ylabel('Memory Saving Ratio')
    plt.savefig(f'./log/pic/{net_type[i]}MultiWorkloadsMSR.png')
    plt.show()

with open('./log/MultiWorkloadsEOR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
x = ['x1', 'x2', 'x3']
for i in range(5):
    TENSILE = df[i, :]
    vDNN = df[i + 5, :]
    Capuchin = df[i + 10, :]
    plt.plot(x, TENSILE)
    plt.plot(x, vDNN)
    plt.plot(x, Capuchin)
    plt.legend(method_type)
    plt.xlabel('workload number')
    plt.ylabel('Extra Overhead Ratio')
    plt.savefig(f'./log/pic/{net_type[i]}MultiWorkloadsEOR.png')
    plt.show()

with open('./log/MultiWorkloadsCBR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
x = ['x1', 'x2', 'x3']
for i in range(5):
    TENSILE = df[i, :]
    vDNN = df[i + 5, :]
    Capuchin = df[i + 10, :]
    plt.plot(x, TENSILE)
    plt.plot(x, vDNN)
    plt.plot(x, Capuchin)
    plt.legend(method_type)
    plt.xlabel('workload number')
    plt.ylabel('Cost-Benefit Ratio')
    plt.savefig(f'./log/pic/{net_type[i]}MultiWorkloadsCBR.png')
    plt.show()

with open('./log/BatchsizeMSR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
x = ['2', '4', '8', '16', '32']
for i, net in enumerate(net_type):
    plt.plot(x, list(df[i]))
plt.legend(net_type)
plt.xlabel('batch size')
plt.ylabel('Memory Saving Ratio')
plt.savefig('./log/pic/BatchSizeMSR.png')
plt.show()

with open('./log/BatchsizeEOR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
# df.index =['2', '4', '8', '16', '32']
# df.to_csv('./log/BatchsizeMSR_temp.csv')
x = ['2', '4', '8', '16', '32']
for i, net in enumerate(net_type):
    plt.plot(x, list(df[i]))
plt.legend(net_type)
plt.xlabel('batch size')
plt.ylabel('Extra Overhead Ratio')
plt.savefig('./log/pic/BatchSizeEOR.png')
plt.show()

with open('./log/BatchsizeCBR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
# df.index =['2', '4', '8', '16', '32']
# df.to_csv('./log/BatchsizeMSR_temp.csv')
x = ['2', '4', '8', '16', '32']
for i, net in enumerate(net_type):
    plt.plot(x, list(df[i]))
plt.legend(net_type)
plt.xlabel('batch size')
plt.ylabel('Cost Benefit Ratio')
plt.savefig('./log/pic/BatchSizeCBR.png')
plt.show()
