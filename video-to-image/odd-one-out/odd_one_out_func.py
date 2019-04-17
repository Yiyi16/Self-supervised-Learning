import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from tqdm import tqdm

data_path = ""


## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

## --------------------- Contrastive Loss ------------------ ##
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1, norm=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.norm = norm

    def forward(self, output1, output2, label):
        dis = F.pairwise_distance(output1, output2, self.norm)
        loss = torch.mean((1-label)*dis+label*torch.clamp(self.margin-dis,min=0.0))
        return loss

## ---------------------- Dataloaders ---------------------- ##
# for CRNN
class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, folders, labels, frames1, frames2, frames3, transform=None):
        "Initialization"
       
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames1 = frames1[:-1]
        self.frames2 = frames2[:-1]
        self.frames3 = frames3[:-1]
        

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, data_path, selected_folder, negfolder, number, use_transform):
        X_1,X_2,X_3 = [],[],[]
        O_1,O_2,O_3=[],[],[]
        XX,YY,ZZ=[],[],[]
        
            
        for i in self.frames1:
            image = Image.open(os.path.join(data_path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            if use_transform is not None:
                image = use_transform(image)
            X_1.append(image)
            #X_1=X_1+(5-2*i)*image
        #print(len(X_1))
        
        for i in self.frames2:
            image = Image.open(os.path.join(data_path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            if use_transform is not None:
                image = use_transform(image)
            X_2.append(image)
             #X_2=X_2+(5-2*i)*image

        
        for i in self.frames3:
            image = Image.open(os.path.join(data_path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            if use_transform is not None:
                image = use_transform(image)

            X_3.append(image)

        
        #print(len(rev))
        X_1 = torch.stack(X_1, dim=0)
        #print (X_1.size())
        X_2 = torch.stack(X_2, dim=0)
        X_3 = torch.stack(X_3, dim=0)
        O_1=5*X_1[0]
        for i in range(len(X_1)-1):
            O_1=O_1+(5-2*(i+1))*X_1[i+1]
            #print("done")
        #O_2=X_2[0]
        O_2=(5-2*0)*X_2[2]
        O_2=O_2+(5-2*1)*X_2[1]
        O_2=O_2+(5-2*2)*X_2[3]
        O_2=O_2+(5-2*3)*X_2[0]
        O_3=5*X_3[0]
        for i in range(len(X_3)-1):
            O_3=O_3+(5-2*(i+1))*X_3[i+1]

        XX=torch.stack((O_1,O_2,O_3),dim=0)
        YY=torch.stack((O_2,O_1,O_3),dim=0)
        ZZ=torch.stack((O_1,O_3,O_2),dim=0)

        return XX,YY,ZZ



    def __getitem__(self, index ):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index].split('__')[0]
        #print(folder)
        randidx = np.random.randint(len(self.folders),size=1)[0]
        negfolder = self.folders[randidx].split('__')[0]
        number = self.folders[index].split('__')[1]
        #print(number)
        
        # Load data
        X,Y,Z = self.read_images(data_path, folder, negfolder, number, self.transform)                  # (input) spatial images
        #y = torch.from_numpy(np.array(self.labels[index])).type(torch.LongTensor)   # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        # ----- time arrow ------ 
       
        
        if index % 3 == 0:
            return X, torch.tensor(1).long()
            
        elif index % 3 == 1:
            return Y, torch.tensor(0).long()
            
        else:
            return Z, torch.tensor(2).long()
## ---------------------- end of Dataloaders ---------------------- ##



class proxyEncoder(nn.Module):
	def __init__(self):
		super(proxyEncoder,self).__init__()

		self.feature=nn.Sequential(
			nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
			nn.ReLU(inplace=True),
			nn.AvgPool2d(kernel_size=3,stride=2),

			nn.Conv2d(64,192,kernel_size=5,stride=1,padding=2),
			nn.ReLU(inplace=True),
			nn.AvgPool2d(kernel_size=3,stride=2),

			nn.Conv2d(192,384,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=True),
		

			nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=True),
		

			nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=True),
			nn.AvgPool2d(kernel_size=3,stride=2),
		)

		self.FC_6=nn.Sequential(
			nn.Dropout(),
			nn.Linear(256*6*6,4096),
			nn.ReLU(inplace=True),
		)

		self.FC=nn.Sequential(
			nn.Dropout(),
			nn.Linear(4096*3,2048),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(2048,3),
		)


	def forward(self,x):
                
                #print("FFFFFFFFFFFF:",x.size())
		a=self.feature(x[:,0])
                #print("a!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:",a.size())
		a=a.view(a.size(0),256*6*6);
			# wait for the image size to adjust the paras!!!!!!!!!!
		a=self.FC_6(a)
                
			
		b=self.feature(x[:,1])
		b=b.view(b.size(0),256*6*6);
			# wait for the image size to adjust the paras!!!!!!!!!!
		b=self.FC_6(b)
			
		c=self.feature(x[:,2])
		c=c.view(c.size(0),256*6*6);
			# wait for the image size to adjust the paras!!!!!!!!!!
		c=self.FC_6(c)

		x=torch.cat((a,b,c),1)
		x=self.FC(x)
		return x


