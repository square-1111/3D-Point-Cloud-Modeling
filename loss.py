from loss_utils import approxMatch, matchCost
import torch

def EMDBatch(X_batch,Y_batch):
    """
    Earth Mover distance between two batches of pointcloud
    """
    def EMD(X,Y):
        """
        Calculates the Earth Mover Distance between two point clouds 
        using Auction LAP

        d_{EMD}(S_1, S_2) = \min_{\phi: S_1 \rightarrow S_2} \sum_{x \in S_1} ||x-\phi(x)|| \quad \textrm{where} \ \phi : S_1 \rightarrow S_2 \ \textrm{is bijection}

        Step 1) Calculate the approximate match matrix between 3D point cloud
        Step 2) Calculate the cost function based on match matrix

        Parameter : 
        X (torch.tensor) : 3D point cloud with dimension (2048,3)
        Y (torch.tensor) : 3D point cloud with dimension (2048,3)

        Returns:
        (int) : Earth mover distance between 3D point clouds X and Y

        Reference:
        1) https://github.com/optas/latent_3d_points/blob/master/external/structural_losses/approxmatch.cpp
        2) https://github.com/daerduoCarey/PyTorchEMD
        
        """
        mat = approxMatch(X, Y)
        cost = matchCost(X, Y, mat)

        return cost

    loss = [EMD(X_batch[i],Y_batch[i]) for i in range(X_batch.shape[0])]
    return loss

############################################################################################

def chamferLossBatch(X_batch,Y_batch):
  """
  Chamfer loss between two batch of point cloud
  """
  def chamferDistance(X, Y):
    """
    Calcuate Chamfer Loss between two point clouds.

    d_{CH}(S_1, S_2)=\sum_{x\in S_1}\min_{y \in S_2}\|x-y\|^2_2 + \sum_{y\in S_2}\min_{x \in S_1}\|x-y\|^2_2

    Step 1) Calcuate the distance matrix between point cloud X and Y
    Step 2) a) Find the coordinates of the points in point cloud Y 
              corresponding to nearest to point cloud X
            b) Find the coordinates of the points in point cloud X 
              corresponding to nearest to point cloud Y
    Step 3) Calculate norm and sum distances
    """

    dist_mat = torch.cdist(X, Y)
    x_min_idx = dist_mat.min(1).indices
    y_min_idx = dist_mat.min(0).indices

    cham_y = ((Y - X[x_min_idx])**2).sum()
    cham_x = ((X - Y[y_min_idx])**2).sum()

    return (cham_y + cham_x)/2048

  loss = 0
  for i in range(X_batch.shape[0]):
    loss += chamferDistance(X_batch[i],Y_batch[i])
  return loss