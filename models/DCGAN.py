import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, final_conv_size, output_channels):
        super(Generator, self).__init__()
        self.neuralnet = nn.Sequential(
            # First step: Project and reshape the initial noise vector
            nn.Linear(noise_dim, final_conv_size * 8 * 4 * 4),
            nn.Unflatten(1, (final_conv_size * 8, 4, 4)),
            nn.ReLU(), 
            nn.BatchNorm2d(final_conv_size * 8),
            
            # 3 Intermediate layers consisting of fractionally-strided convolutions, ReLU activation and batch normalisation
            self.layer(final_conv_size * 8, final_conv_size * 4),

            self.layer(final_conv_size * 4, final_conv_size * 2),

            self.layer(final_conv_size * 2, final_conv_size),

            # Final layer consistinf of fractionally-strided convolution and Tanh activation function
            nn.ConvTranspose2d(final_conv_size, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def layer(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, input):
        return self.neuralnet(input)