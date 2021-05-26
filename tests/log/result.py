import datetime
res = open('./result.txt', 'w')
with open('type0_VGG16_net_order=1_record_2.txt','r') as f:
    lines = f.readlines()
vanilla_max_memory = 0
for line in lines:
    memory = float(line.split('\t')[1].split(' ')[1])
    if memory>vanilla_max_memory:
        vanilla_max_memory = memory
# vanilla_max_memory *= 2
with open('type0_VGG16_net_order=1_record_1.txt','r') as f:
    lines = f.readlines()
vanilla_time = 0
for line in lines:
    t = line.split('\t')[0].split(' ')[3]
    t = datetime.datetime.strptime(t, '%H:%M:%S.%f').second*1000000 + datetime.datetime.strptime(t, '%H:%M:%S.%f').microsecond
    vanilla_time+=t
# vanilla_time /= 2
res.writelines('vanilla:\n')
res.writelines(f'max_memory:{vanilla_max_memory}\n')
res.writelines(f'time:{vanilla_time}\n\n')

with open('type1_VGG16_net_order=1_record_2.txt','r') as f:
    lines = f.readlines()
max_memory = 0
for line in lines:
    memory = float(line.split('\t')[1].split(' ')[1])
    if memory>max_memory:
        max_memory = memory
with open('type1_VGG16_net_order=1_record_1.txt','r') as f:
    lines = f.readlines()
time = 0
for line in lines:
    t = line.split('\t')[0].split(' ')[3]
    # t = datetime.datetime.strptime(t, '%H:%M:%S.%f').microsecond
    t = datetime.datetime.strptime(t, '%H:%M:%S.%f').second * 1000000 + datetime.datetime.strptime(t,'%H:%M:%S.%f').microsecond
    time+=t
memory_saved = 1-max_memory/vanilla_max_memory
extra_overhead = 1-vanilla_time/time
res.writelines('capuchin:\n')
res.writelines(f'max_memory:{max_memory}\n')
res.writelines(f'time:{time}\n')
res.writelines(f'memory_saved:{memory_saved}\n')
res.writelines(f'extra_overhead:{extra_overhead}\n')
res.writelines(f'efficiency:{memory_saved/extra_overhead}\n\n')


with open('type2_VGG16_net_order=1_record_2.txt','r') as f:
    lines = f.readlines()
max_memory = 0
for line in lines:
    memory = float(line.split('\t')[1].split(' ')[1])
    if memory>max_memory:
        max_memory = memory
with open('type2_VGG16_net_order=1_record_1.txt','r') as f:
    lines = f.readlines()
time = 0
for line in lines:
    t = line.split('\t')[0].split(' ')[3]
    # t = datetime.datetime.strptime(t, '%H:%M:%S.%f').microsecond
    t = datetime.datetime.strptime(t, '%H:%M:%S.%f').second * 1000000 + datetime.datetime.strptime(t,'%H:%M:%S.%f').microsecond
    time+=t
memory_saved = 1-max_memory/vanilla_max_memory
extra_overhead = 1-vanilla_time/time
res.writelines('vDNN:\n')
res.writelines(f'max_memory:{max_memory}\n')
res.writelines(f'time:{time}\n')
res.writelines(f'memory_saved:{memory_saved}\n')
res.writelines(f'extra_overhead:{extra_overhead}\n')
res.writelines(f'efficiency:{memory_saved/extra_overhead}\n\n')

