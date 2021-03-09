def load_gpu():
    with open('../../GPU.txt') as f:
        c = f.readlines()
        c = c.strip()
        GPU = int(c)
    return GPU