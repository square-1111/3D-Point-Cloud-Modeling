import torch

def approxMatch(X, Y):
    """
    Calculates Approximate Matching. An iterative algorithm to calculate 
    matching matrix.

    Parameter : 
    X (torch.tensor) : 3D point cloud of dimension (2048,3)
    Y (torch.tensor) : 3D point cloud of dimension (2048,3)

    Returns:
    (torch.tensor) : A matching matrix of dimension (2048, 2048) 
    """

    n = X.shape[0]
    m = Y.shape[0]
    factorl = max(n,m)/n
    factorr = max(n,m)/m

    device = X.get_device()

    saturatedl = torch.ones(n,dtype=torch.float).to(device)*factorl
    saturatedr = torch.ones(m,dtype=torch.float).to(device)*factorr

    match = torch.zeros((n,m),dtype=torch.float).to(device)

    for i in range(7,-3,-1):
        level = -torch.pow(torch.tensor(4.0),torch.tensor(i)).to(device)
        if i == -2:
            level = torch.tensor(0, dtype=torch.float).to(device)
    
        
        weight = torch.exp(level*torch.cdist(X, Y)*torch.cdist(X, Y))*saturatedr
        # print(weight)

        s = torch.sum(weight, axis=1)
        s[s < 1e-9] = 1e-9

        weight = weight/s[:,None]*saturatedl[:,None]
        ss = torch.ones(n,dtype=torch.float).to(device)*1e-9
        ss += torch.sum(weight, axis=0)
        ss[ss < 1e-9] = 1e-9

        ss = saturatedr/ss 
        ss[ss>1.0] = 1.0

        weight = weight*ss
        s = torch.sum(weight, axis=0)
        ss2 = torch.sum(weight, axis=0)

        saturatedl = saturatedl-s
        saturatedl[saturatedl < 0] = 0

        match = match + weight

        saturatedr = saturatedr - ss2
        saturatedr[saturatedr < 0] = 0

    return match

def matchCost(X, Y, match):
    """
    Calculates Loss

    Parameter : 
    X (torch.tensor) : 3D point cloud of dimension (2048,3)
    Y (torch.tensor) : 3D point cloud of dimension (2048,3)
    (torch.tensor) : A matching matrix of dimension (2048, 2048) from
                    approxMatch function.

    Returns:
    (int) : Earth Mover Distance 
    """ 
    
    cost = (torch.cdist(X, Y)*match).sum()
    return cost