import os
import torch
import torchvision
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.vae import LVAE
import numpy as np

matplotlib.style.use('ggplot')
directory1 = "samples"
directory2 = "generated"
parent_dir = "./"
path1 = os.path.join(parent_dir, directory1) 
path2 = os.path.join(parent_dir, directory2) 
os.mkdir(path1)
os.mkdir(path2)
#parser = argparse.ArgumentParser()
#parser.add_argument('-e','--epochs',default=10,type=int)
#args = vars(parser.parse_args())

def floss(loss, mu, log_var):

    LOSS = loss
    KLD  = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return LOSS + KLD


def main():
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train = torchvision.datasets.MNIST(
        root = './data',
        train = True,
        download = True,
        transform=transform
    )

    val = torchvision.datasets.MNIST(
        root = './data',
        train = False,
        download = True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True)

    model = LVAE(28*28)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)
    criterion = nn.BCELoss(reduction='sum')

    training_loss = []
    validation_loss = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for i, data in tqdm(enumerate(train_loader), total=int(len(train)//train_loader.batch_size)):
            inputs, _ = data
            inputs = inputs.to(DEVICE)
            inputs = inputs.view(inputs.size(0),-1)
            optimizer.zero_grad()

            out, mu, log_var = model.forward(inputs)
            reconstruction_loss = criterion(out,inputs)
            loss = floss(reconstruction_loss,mu,log_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                num_rows = 8
                both = torch.cat((inputs.view(BATCH_SIZE,1,28,28)[:8],
                                   out.view(BATCH_SIZE,1,28,28)[:8]
                 ))
                torchvision.utils.save_image(both.cpu(),f'./samples/{epoch}_{i}.png',nrow=num_rows)

                z = np.random.normal(0,1,(64,16))
                z = torch.from_numpy(z).to(DEVICE)
                z = z.float()
                new = model.decode(z)
                new = new.view(BATCH_SIZE,1,28,28)[:8]
                torchvision.utils.save_image(new.cpu(),f'./generated/{epoch}_{i}.png',nrow=4)



        train_loss = total_loss/(len(train_loader.dataset))

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader), total = int(len(val)/val_loader.batch_size)):
                inputs, _ = data
                inputs = inputs.to(DEVICE)
                inputs = inputs.view(inputs.size(0), -1)

                out, mu, log_var = model.forward(inputs)
                reconstruction_loss = criterion(out,inputs)
                loss = floss(reconstruction_loss, mu, log_var)
                total_loss += loss.item()

        val_loss = total_loss/(len(val_loader.dataset))

        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")





if __name__=="__main__":
    main()
