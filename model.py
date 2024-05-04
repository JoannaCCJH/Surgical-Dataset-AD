import torch

# Creating a DeepAutoencoder class 
class DeepAutoencoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()         
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(256 * 256*3, 512), 
            torch.nn.ReLU(), 
            torch.nn.Linear(512, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 64) 
        ) 
          
        self.decoder = torch.nn.Sequential( 
            torch.nn.Linear(64, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 512), 
            torch.nn.ReLU(), 
            torch.nn.Linear(512, 256 * 256*3), 
            torch.nn.Sigmoid() 
        ) 
  
    def forward(self, x): 
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        return encoded, decoded 
    
class ConvDeepAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvDeepAutoencoder, self).__init__()

        # Encoder layers
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # input channels: 3 (RGB), output channels: 16, kernel_size: 3x3, stride: 2
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # input channels: 16, output channels: 32, kernel_size: 3x3, stride: 2
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # input channels: 32, output channels: 64, kernel_size: 3x3, stride: 2
            torch.nn.ReLU()
        )

        # Decoder layers
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # input channels: 64, output channels: 32, kernel_size: 3x3, stride: 2
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # input channels: 32, output channels: 16, kernel_size: 3x3, stride: 2
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # input channels: 16, output channels: 3, kernel_size: 3x3, stride: 2
            torch.nn.Sigmoid() # Sigmoid activation to ensure output values are in [0, 1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
if __name__ == "__main__":
    
    # Instantiating the model and hyperparameters 
    model = DeepAutoencoder() 
    criterion = torch.nn.MSELoss() 
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 