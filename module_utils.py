def build_model_fully_conected(input_dim, output_dim, hidden_dim, num_layers, activation_func=ReLU):
    """
    Builds a PyTorch neural network model.

    Parameters:
    -----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output features.
    hidden_dim : int
        Number of units in each hidden layer.
    num_layers : int
        Number of hidden layers in the network (excluding the first input layer)
    activation_func : callable, optional (default=ReLU)
        Activation function to apply after each hidden layer.

     Returns:
    --------
    a Fully conected network
    """

      layers = []

      layers.append(nn.Linear(input_dim, hidden_dim))
      layers.append(nn.activation_func())

      for _ in range(num_layers):
          layers.append(nn.Linear(hidden_dim, hidden_dim))
          layers.append(nn.activation_func())

      layers.append(nn.Linear(hidden_dim, output_dim))

      model = nn.Sequential(*layers)
      return model


import torch.nn as nn

def build_model_conv(
    in_channels,
    out_channels,
    hidden_channels=[512, 256, 128, 64],
    kernel_size=4,
    stride=2,
    padding=1,
    hidden_activation=nn.ReLU,
    output_activation=nn.Sigmoid
):
    """
    Builds a flexible ConvTranspose2d (upsampling) network.

    Parameters:
    -----------
    in_channels : int
        Number of input channels for the first layer.
    out_channels : int
        Number of output channels for the last layer.
    hidden_channels : list of int
        Number of channels for each intermediate ConvTranspose2d layer.
    kernel_size : int
        Kernel size for all ConvTranspose2d layers.
    stride : int
        Stride for all ConvTranspose2d layers.
    padding : int
        Padding for all ConvTranspose2d layers.
    hidden_activation : callable
        Activation function applied after each hidden layer.
    output_activation : callable
        Activation function applied after the last layer.

    Returns:
    --------
    torch.nn.Sequential
        A PyTorch sequential model consisting of ConvTranspose2d layers.
    """
    layers = []
    channels = [in_channels] + hidden_channels

    # Add hidden ConvTranspose2d layers
    for i in range(len(hidden_channels)):
        layers.append(nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size, stride, padding))
        layers.append(hidden_activation(inplace=True))

    # Add final output layer
    layers.append(nn.ConvTranspose2d(hidden_channels[-1], out_channels, kernel_size, stride, padding))
    if output_activation is not None:
        layers.append(output_activation())

    return nn.Sequential(*layers)
