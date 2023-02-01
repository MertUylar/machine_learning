from  PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
kedi=np.array(Image.open("indir.png").resize((224,224)))

print(kedi)

keditorch=torch.from_numpy(kedi)
print(np.shape(kedi)) # size ile aynı şey
print(keditorch.size())


# plt.imsave("kedi2",keditorch) 
