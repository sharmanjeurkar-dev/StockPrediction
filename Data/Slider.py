import numpy as np
class Slider():
    
    def __init__(self,feature:np.array,labels:np.array,length:int):
        self.feature = feature
        self.labels = labels
        self.length = length



    def slider(self):
        x = []
        y = []

        for i in range(len(self.feature) - self.length):
            xws = self.feature[i:(i+self.length)]
            yws = self.labels[(i+self.length)]

            x.append(xws)
            y.append(yws)

        return np.array(x),np.array(y)