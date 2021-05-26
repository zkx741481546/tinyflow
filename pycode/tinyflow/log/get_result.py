workload = 'VGG x2 new'
vanilla_path = f'./{workload}/vanilla/'
scheduled_path = f'./{workload}/scheduled/'
with open(vanilla_path+'gpu_record.txt', 'r') as f:
    lines = f.readlines()
temp = lines[-1].split('\t')
vanilla_max_memory_used = float(temp[2].split(' ')[1])
with open(scheduled_path+'gpu_record.txt', 'r') as f:
    lines = f.readlines()
temp = lines[-1].split('\t')
schedule_max_memory_used = float(temp[2].split(' ')[1])
saved_ratio = 1-schedule_max_memory_used/vanilla_max_memory_used
with open(vanilla_path+'gpu_time.txt', 'r') as f:
    lines = f.readlines()
vanilla_time_cost = float(lines[0].replace('time_cost:', ''))
with open(scheduled_path+'gpu_time.txt', 'r') as f:
    lines = f.readlines()
schedule_time_cost = float(lines[0].replace('time_cost:', ''))
extra_overhead = 1-vanilla_time_cost/schedule_time_cost

memory_saved_to_extra_overhead_ratio = saved_ratio/extra_overhead
with open(f'./{workload}/result.txt', 'w') as f:
    f.write(f'saved_ratio:{saved_ratio}'
            f'\nextra_overhead:{extra_overhead}'
            f'\nvanilla_max_memory_used:{vanilla_max_memory_used}'
            f'\nschedule_max_memory_used:{schedule_max_memory_used}'
            f'\nvanilla_time_cost:{vanilla_time_cost}'
            f'\nschedule_time_cost:{schedule_time_cost}'
            f'\nefficiency:{memory_saved_to_extra_overhead_ratio}')
