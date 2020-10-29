import torch
import torch.nn as nn


class LVAE(nn.Module):
    def __init__(self, in_features):
        super(LVAE, self).__init__()
        self.features = 16

        self.fc1 = nn.Linear(in_features,512)
        self.fc2 = nn.Linear(512,self.features*2)

        self.fc3 = nn.Linear(self.features,512)
        self.fc4 = nn.Linear(512,in_features)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #print('Before view',x.shape)
        x = x.view(-1, 2, self.features)
        #print('After view',x.shape)

        mu = x[:,0,:]
        log_var = x[:,1,:]

        z = self.reparameterize(mu, log_var)
        #print(z.shape)
        x = self.fc3(z)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        #print('out view',x.shape)

        return x, mu, log_var

    def decode(self,z):
        z = self.fc3(z)
        z = self.relu(z)
        z = self.fc4(z)
        z = self.sigmoid(z)
        #print('out view',x.shape)

        return z
