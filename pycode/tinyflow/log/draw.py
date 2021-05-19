import numpy as np
import matplotlib.pyplot as plt

with open('./gpu_record_no_swap.txt') as f:
    lines = f.readlines()
old_memory = []
for line in lines:
    temp = line.split('\t')
    m = temp[3].split(' ')[1]
    old_memory.append(m)
with open('./gpu_record_swap.txt') as f:
    lines = f.readlines()
memory = []
for line in lines:
    temp = line.split('\t')
    m = temp[3].split(' ')[1]
    memory.append(m)
if len(memory)<len(old_memory):
    old_memory = old_memory[:len(memory)]
else:
    memory = memory[:len(old_memory)]
x = np.arange(len(memory))
plt.plot(x,old_memory)
plt.plot(x,memory)
plt.show()