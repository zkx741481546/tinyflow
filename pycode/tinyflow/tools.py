def load_gpu():
    with open('../../GPU.txt') as f:
        c = f.readlines()
        c = c[0].strip()
        GPU = int(c)
    return GPU