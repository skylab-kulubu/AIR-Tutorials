#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:06:58 2020

@author: safak
"""
#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(    
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.fc = nn.Linear(7*7*32,10)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1,7*7*32)
        out = self.fc(out)
        return out


def show_im(col,row,train):
    fig = plt.figure(figsize=(8,8))
    for i in range(1,col*row + 1):
        rand_label = np.random.randint(len(train))
        img = train[rand_label][0][0,:,:]
        fig.add_subplot(row,col,i)
        plt.title(labels_map[train_set[rand_label][1]])
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.show()
#%%


if __name__ == "__main__":
    
    train_set = datasets.FashionMNIST(root='/media/safak/Data/Desktop HDD/Deep Learning/PyTorch/CNN',
                                      train = True,
                                      download = True,
                                      transform = transforms.Compose([transforms.ToTensor()]))
    test_set = datasets.FashionMNIST(root='/media/safak/Data/Desktop HDD/Deep Learning/PyTorch/CNN',
                                     train = False,
                                     download = False,
                                     transform = transforms.Compose([transforms.ToTensor()]))
    labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}
#%%    
    show_im(4,5,train_set)
#%%
    batch_size = 100
    lr = 0.001
    epochs = 10
    
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size = batch_size,
                                               shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size = batch_size,
                                              shuffle = True)
    
    model = CNN()
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
#%%
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    for epoch in range(epochs):
        model.train()
        correct = 0
        iterations = 0
        each_loss = 0.0
        for i,(inputs,targets) in enumerate(train_loader):
            if CUDA:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            each_loss+=loss.item()
            _, predicted=torch.max(outputs,1)
            correct += (predicted == targets).sum().item()
            iterations += 1
        train_loss.append(each_loss/iterations)
        train_accuracy.append(100 * correct/len(train_set))
        print("Epoch: {}/{}".format(epoch+1,epochs))
        print("Training Loss: {:.4f}".format(train_loss[-1]))
        print("Training Accuracy: {:.4f}".format(train_accuracy[-1]))
        model.eval()
        correct = 0
        iterations = 0
        each_loss = 0.0
        for i,(inputs,targets) in enumerate(test_loader):
            if CUDA:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model.forward(inputs)
            loss = criterion(outputs,targets)
            each_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            correct += (predicted == targets).sum().item()
            iterations += 1
        test_loss.append(each_loss/iterations)
        test_accuracy.append(100 * correct / len(test_set))
        print("Test Loss: {:.4f}".format(test_loss[-1]))
        print("Test Accuracy: {:.4f}".format(test_accuracy[-1]))
        print(48*"_")
#%%
    fig = plt.figure(figsize=(10,10))
    plt.plot(train_loss,label = 'Train Loss')
    plt.plot(test_loss,label='Test Loss')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(train_accuracy,label = 'Train Accuracy')
    plt.plot(test_accuracy,label='Test Accuracy')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()
#%%
    import cv2 
    from PIL import Image
    transforms_photo = transforms.Compose([transforms.ToTensor()])
    def predict_yours(img_name: str, model):
        img = cv2.imread(img_name,0)
        img = cv2.resize(img, (28,28))
        ret, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
        # img = 255 - img
        plt.imshow(img,cmap='gray')
        img = Image.fromarray(img)
        img = transforms_photo(img)
        img = img.view((1,1,28,28))
        
        model.eval()
        if CUDA:
            model = model.cuda()
            img = img.cuda()
        out = model.forward(img)
        print(out)
        print(out.data)
        _, predd = torch.max(out,1)

        return predd.item()
    pred = predict_yours('jeans9.png', model)
    print("The number is: ",pred)
#%%
    
    def plot_kernels(tensor, num_cols=6):
        num_kernels = tensor.shape[0]
        num_rows = 1+ num_kernels // num_cols
        fig = plt.figure(figsize=(num_cols,num_rows))
        for i in range(num_kernels):
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            ax1.imshow(tensor[i][0,:,:], cmap='gray')
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    filters = model.modules();
    model_layers = [i for i in model.children()]
    first_layer = model_layers[0]
    second_layer = model_layers[1]
    first_kernels = first_layer[0].cpu().weight.data.numpy()
    plot_kernels(first_kernels, 8)
    second_kernels = second_layer[0].cpu().weight.data.numpy()
    plot_kernels(second_kernels, 8)
