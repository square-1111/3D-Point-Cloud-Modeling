import numpy as np
import yaml, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from IPython import display
import torch

from make_data import Data

### Loading Configuration
def parseYAML(yaml_file):
    """
    A function to parse YAML file, holding configuration

    Args:
    file (yaml) : Path to yaml file

    Returns:
    dict_ : Dictionary containing configuration
    """
    with open(yaml_file) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

### Function to generate noise vector
def noiseFunc(mu, sigma, batch_size, device):
    """
    Function to generate noise from Gaussian Distribution

    Args:
    mu, sigma : Mean and Standard Deviation for Normal Distribution
    device    : GPU or CPU device
    """
    return torch.normal(mu, sigma, (batch_size, 128)).float().to(device)

### Loss Visualisation
def plotLoss(losses, model_name, path='./'):
    """
    A function to visualize loss to see changes in loss

    Args:
    losses (dict)   : A list of dictionary
    model_name(str) : Name of model along with criterion
    path(str)       : Path where to save loss visualization
    """

    loss_path = path + model_name
    display.clear_output(wait=True)
    display.display(plt.gcf())
    for i in losses:
        plt.figure(figsize=(10,8))
        plt.plot(losses[i], label=i)
        plt.title(i)

    plt.legend()
    plt.savefig('{}loss.png'.format(loss_path))
    plt.show()

### Plot Point Cloud samples
def plotPointCloud(object, model=None):
    """
    Retrieve 4 random point clouds and plot interactive graphs  
    """

    # Layout for plot
    fig = make_subplots(
        rows=2, cols=2,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
    
    objlst = Data(object)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    for i in range(4):
        point_cloud = objlst[np.random.randint(len(objlst))]

        if model is not None and 'Autoencoder' in model.name:
            point_cloud = model(point_cloud[None,:].to(device))
            point_cloud = torch.squeeze(point_cloud,0)
        
        if model is not None and 'Generator' in model.name:
            noise = noiseFunc(0, 0.2, 1, device)
            point_cloud = model(noise)
            point_cloud = torch.squeeze(point_cloud,0)


        np_point_cloud = point_cloud.detach().cpu().numpy()
        
        fig.add_trace(
            go.Scatter3d(x=np_point_cloud[:,0], 
                        y=np_point_cloud[:,1], 
                        z=np_point_cloud[:,2],
                        mode='markers',
                        marker=dict(size=1,
                                    color=np_point_cloud[:,2],                
                                    colorscale='Viridis',   
                                    opacity=0.8
                                    )),
                        row=i//2 + 1, col=i%2 + 1
                        )
      
    fig.update_layout(
        showlegend=False,
        height=800,
        width=1500
        )

    fig.show()


### Save Model
def saveModel(path, model, epoch):
    """
    Save the model and its weight to specified path
    
    Attributes:
    path : location where to save model
    model : weighted model
    epoch : save epoch every 10 epochs
    """
    path = path + str(epoch)
    torch.save(model.state_dict(), path)

### Check Loop
def currEpoch(path):
    """
    Check the current epoch number. This is to continue training from the 
    current epoch number

    Args:
    path : path of model
    """
    list_dir = os.listdir(path)
    
    if len(list_dir) == 0:
        return -1
    
    else:
        list_dir = [int(i) for i in list_dir]
        return max(list_dir)
