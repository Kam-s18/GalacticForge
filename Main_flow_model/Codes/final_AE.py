#Import librarires
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

def AutoEncoder(dataset_folder, num_epochs=100, batch_size=4, latent_dim=16, learning_rate=0.00003, model_save_path='AE_Galaxy_final.pth', latent_space_save_path='final_latent_spaces'):

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader for Galaxy Images
    class GalaxyImages(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.images = os.listdir(root_dir)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.images[idx])
            image = Image.open(img_name).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
        
    #DEEP AUTOENCODER
    class AE_Galaxy(nn.Module):
        def __init__(self, latent_dim=latent_dim):
            super(AE_Galaxy, self).__init__()
            self.latent_dim = latent_dim
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 3x448x448 -> 64x224x224
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x224x224 -> 128x112x112
                nn.ReLU(True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128x112x112 -> 256x56x56
                nn.ReLU(True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 256x56x56 -> 512x28x28
                nn.ReLU(True),
                nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 512x28x28 -> 1024x14x14
                nn.ReLU(True),
                nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),  # 1024x14x14 -> 2048x7x7
                nn.ReLU(True),
                nn.Flatten(),  # Flatten for fully connected layer
                nn.Linear(2048*7*7, self.latent_dim)  # Compress to latent space
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 2048*7*7),  # Expand from latent space
                nn.Unflatten(1, (2048, 7, 7)),  # Unflatten to 4D
                nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # 2048x7x7 -> 1024x14x14
                nn.ReLU(True),
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 1024x14x14 -> 512x28x28
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 512x28x28 -> 256x56x56
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 256x56x56 -> 128x112x112
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x112x112 -> 64x224x224
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 64x224x224 -> 3x448x448
                nn.Sigmoid()  # To normalize the output to [0, 1]
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        

    #Gettting the data ready
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))  
    ])

    # Dataset
    dataset = GalaxyImages(dataset_folder, transform=transform)

    # Splitting dataset 
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #Set loss and optimizer
    autoencoder = AE_Galaxy(latent_dim=latent_dim)
    autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Training Loop
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        autoencoder.train()
        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            data = data.to(device)
            recon_batch = autoencoder(data)
            
            loss = criterion(recon_batch, data)
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
    torch.save(autoencoder.state_dict(), model_save_path)

    #DEEP AUTOENCODER
    autoencoder = AE_Galaxy(latent_dim=16)
    autoencoder.to(device)
    autoencoder.load_state_dict(torch.load(r"AE_Galaxy_final.pth"))

    # Testing Loop
    autoencoder.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            recon_batch = autoencoder(data)
            loss = criterion(recon_batch, data)
            test_loss += loss.item()

    test_loss /= len(test_dataloader)
    print(f'Test Loss: {test_loss:.4f}')

    #Saving all latent spaces in numpy files
    image_folder = dataset_folder
    output_folder = latent_space_save_path 

    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    autoencoder.eval()

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        
        transformed_image = transform(image).unsqueeze(0)  
        transformed_image = transformed_image.to(device)  
        
        
        with torch.no_grad():
            latent_space = autoencoder.encoder(transformed_image) 
        
        latent_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_latent.npy")
        np.save(latent_file_path, latent_space.squeeze().cpu().numpy())

    print("Latent space files saved successfully!")

