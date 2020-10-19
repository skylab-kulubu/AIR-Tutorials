#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:55:23 2020

@author: safak
"""

#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # grayscale has 1 channel + 8 filters + 3x3 filter
        # same padding= (filter size - 1)/2 -> (3 -1)/2 = 1 
        # same padding -> input size = output size
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # output size of each feature map = 
        # ((input size - filter size + 2(padding))/(strinde) + 1) -> (28 - 3 + 2)/1 + 1 = 28
        self.batch_norm1 = nn.BatchNorm2d(8) # feature map = 8
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        # the output size will be = 28/2 = 14
        # same padding= (filter size - 1)/2 -> (5-1)/2 = 2
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        # output size of each feature map = 
        # ((input size - filter size + 2(padding))/(stride) + 1) -> (14 - 5 + 2*2)/1 + 1 = 14
        self.batch_norm2 = nn.BatchNorm2d(32)
        """there are a maxpooling again filter size = 2x2"""
        """so output size is now 14/2 = 7 """
        # 32 feature map flatten (output size = 7x7):
        # 7*7*32 = 1568
        self.fc1 = nn.Linear(1568,600)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(600,10)
    
    def forward(self,x):
        out = self.cnn1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.cnn2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # flattening:
        out = out.view(-1,1568) #(batch_size,1568) == (100,1568) == (-1,1568) -1 is flexible
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
#%%
if __name__ == "__main__":
    
    
    mean = 0.1307
    std = 0.3081
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((mean,),(std,))])
    
    train_dataset = datasets.MNIST(root='/media/safak/Data/Desktop HDD/Deep Learning/PyTorch/CNN',
                                   train = True,transform = transforms, download = True)
    test_dataset = datasets.MNIST(root='/media/safak/Data/Desktop HDD/Deep Learning/PyTorch/CNN',
                                   train = False,transform = transforms)
#%%
    random_img = train_dataset[20][0].numpy() * std + mean
    plt.imshow(random_img.reshape(28,28),cmap='gray')
    print(train_dataset[20][1])
#%%
    batch_size = 100
    train_load = torch.utils.data.DataLoader(dataset = train_dataset,
                                             batch_size = batch_size,
                                             shuffle = True)
    test_load = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = batch_size,
                                              shuffle = True)
    
    print(len(train_dataset))
    print(len(train_load))
    print(len(test_dataset))
    print(len(test_load))
    
#%%
    model = CNN()
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)
#%%
    epochs = 10
    training_loss = []
    training_accuracy= []
    test_loss = []
    test_accuracy = []
    
    for epoch in range(epochs):
        correct = 0
        iterations = 0
        each_loss = 0.0
        
        model.train()
        for i,(inputs, targets) in enumerate(train_load):
            if CUDA: 
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            each_loss+=loss.item() # extract the value from tensor
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted=torch.max(outputs,1)
            correct += (predicted == targets).sum().item()
            iterations += 1
        training_loss.append(each_loss/iterations)
        training_accuracy.append(100 * correct/len(train_dataset))
        
        #now we test:
        each_loss_test = 0.0
        iterations = 0 
        correct = 0
        model.eval()
        for i,(inputs, targets) in enumerate(test_load):
            if CUDA: 
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            each_loss_test+=loss.item() # extract the value from tensor
            
            _, predicted=torch.max(outputs,1)
            correct += (predicted == targets).sum().item()
            iterations += 1
        
        test_loss.append(each_loss_test/iterations)
        test_accuracy.append(100 * correct/len(test_dataset))
        print("Epoch {}/{} - Training Loss: {:.3f} - Training Accuracy: {:.3f} - Test Loss: {:.3f} - Test Accuracy: {:.3f}"
              .format(epoch+1,epochs,training_loss[-1],training_accuracy[-1],test_loss[-1],test_accuracy[-1]))
#%% plotting the results
        
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10))
    plt.plot(training_loss,label = 'Train Loss')
    plt.plot(test_loss,label='Test Loss')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(training_accuracy,label = 'Train Accuracy')
    plt.plot(test_accuracy,label='Test Accuracy')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()
#%% predicting an image from test data
    img = test_dataset[45][0].resize_((1,1,28,28))
    label = test_dataset[45][1]
    model.eval()
    if CUDA:
        model = model.cuda()
        label = label.cuda()
        img = img.cuda()
    out = model.forward(img)
    _,predicted = torch.max(out,1)
    print("Predicted:",predicted.item())
    print("Real Value:",label)
    
#%% predict your own hadwritten digit
    import cv2 
    from PIL import Image
    transforms_photo = transforms.Compose([transforms.Resize((28,28)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((mean,),(std,))])
    def predict_yours(img_name: str, model):
        img = cv2.imread(img_name,0)
        ret, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = 255 - threshold
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
    pred = predict_yours('../samples/3.jpg', model)
    print("The number is: ",pred)
