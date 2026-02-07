import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset



@staticmethod
#Converts numpy raw data to tensors
def convertNumpyToTensors(x:np.ndarray,y:np.ndarray):
    xf = torch.from_numpy(x).float()
    yf = torch.from_numpy(y).float()

    if len(yf.shape) == 1:  #redendency to avoid errors with many loss functions which gives error 
        yf = yf.unsqueeze(1)
        print("Unsqueezing executed")

    return xf,yf

@staticmethod
#adding X and Y to TensorDataset
def createTensorDataset(X:torch.Tensor,Y:torch.Tensor):
    dataset = TensorDataset(X,Y)

    return dataset

@staticmethod
#Loading Dataset
def loadData(dataset:torch.utils.data.Dataset,
            batch:int,
            num_worker:int,
            shuffle = True
            ):
    
    dataload = DataLoader( dataset=dataset,
                        batch_size=batch,
                        num_workers=num_worker,
                        shuffle=shuffle
    )

    return dataload
