import torch
import numpy as np
 
if __name__ == '__main__':
    x = np.random.randn(1)
    try:
        t = torch.cuda.FloatTensor(x)
        print('Success!')
    except Exception as e:
        print(e)