import os, torch
import numpy as np
from plyfile import PlyData, PlyElement
import plotly.graph_objects as go
from plotly.subplots import make_subplots

## Mapping of object_id to objects
id_object = {
    '02691156': 'plane',      '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',    '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',    '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',     '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',    '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',        '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',      '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',      '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',      '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',   '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',     '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',      '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',    '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave',  '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',      '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',        '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',      '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',       '04330267': 'stove',      '04530566': 'watercraft',
    '04554684': 'washer',     '02858304': 'boat',       '02992529': 'cellphone'
}

## Mapping object to object_id  
object_id_dict = {v: k for k, v in id_object.items()}

class Data():
    """
    A class to represent the object

    Attributes:
    object (str) : name of object to plot from the given list
                    ['plane', 'bag', 'basket', 'bathtub', 'bed', 'bench',
                    'bicycle', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 
                    'bus', 'cabinet', 'can', 'camera', 'cap', 'car', 'chair', 
                    'clock', 'dishwasher', 'monitor', 'table', 'telephone', 
                    'tin_can', 'tower', 'train', 'keyboard', 'earphone', 
                    'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 
                    'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 
                    'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 
                    'pistol', 'pot', 'printer', 'remote_control', 'rifle', 
                    'rocket', 'skateboard', 'sofa', 'stove', 'watercraft', 
                    'washer', 'boat', 'cellphone']

    """
    def __init__(self, object):
        object_id = object_id_dict.get(object)
        source_dir = 'data/shape_net_core_uniform_samples_2048/'
        self.obj_dir = source_dir + object_id + '/'
        self.list_IDs = os.listdir(source_dir + object_id + '/')
        
        
    
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        file_path = self.obj_dir + self.list_IDs[index]
        plydata = PlyData.read(file_path)
        np_point_cloud = np.array((plydata['vertex']['x'],
                                plydata['vertex']['y'],
                                plydata['vertex']['z'])).T

        return torch.from_numpy(np_point_cloud).float()



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


    for i in range(4):
        point_cloud = objlst[np.random.randint(len(objlst))]

        if model is not None and 'Autoencoder' in model.name:
            point_cloud = model(point_cloud[None,:])
            point_cloud = torch.squeeze(point_cloud,0)
        
        if model is not None and 'Generator' in model.name:
            noise = noiseFunc(0, 0.2, 1)
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