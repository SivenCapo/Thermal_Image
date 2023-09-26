import os
import numpy as np
import imageio
from utils import extract_patches,blur,upscale
from torch import from_numpy,stack

def dataset(dataset_train_path,batch_size,scale_factor):
    assert(os.path.exists(dataset_train_path))
    data = []
    print(1)
    conter=0
    for file in os.listdir(dataset_train_path):
        if file.endswith('.png'):
            filepath = os.path.join(dataset_train_path,file)
            img = imageio.imread(filepath)
            conter+=1
            if conter%100==0:
                print(conter)
            #.dot([0.299, 0.587, 0.114])
            patches = extract_patches(img,(256,256),0.166)

            data += [patches[idx] for idx in range(patches.shape[0])]
    print("number of patches is")
    #print(data[1])
    conter=0
    # for patch in data:
    #     mod_data = from_numpy(np.expand_dims(blur(upscale(patch,scale_factor),scale_factor),0)).float()
    #     conter+=1
    #     if conter%100==0:
    #         print(conter)
    mod_data = [from_numpy(np.expand_dims(blur(upscale(patch,scale_factor),scale_factor),0)).float() for patch in data]
    print("blured data generated")
    data = [from_numpy(np.expand_dims(upscale(patch,scale_factor),0)).float() for patch in data]
    #print(type(data[2]))
    print("gt data generated")
    l = len(data)
    for idx in range(0,l,batch_size):
        data_batch = data[idx]
        mod_batch= mod_data[idx]
        yield mod_batch, data_batch


