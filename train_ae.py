import os, torch
from utils import currEpoch, saveModel, plotLoss, plotPointCloud

def trainingLoopAE(obj, model, num_epoch, train_dataloader, criterion, optimizer, loss_dict, path, device):
    """
    A universal training loop to optimize any loss function

    Args:
    
    obj       : Object on which model is to be trained
    model     : Architecture of the neural network
    num_epoch : number of iterations
    criterion : loss function which we need to optimize
    train_dataloader : Training Dataset
    optimizer : optimize the loss using this optimizer
    loss_dict : A dictionary to keep track of loss
    path      : location where to store model
    device    : GPU or CPU device
    """

    # Check if path to directory exist. If no: then create one
    if not os.path.exists(path):
        os.makedirs(path)

    curr_epoch = currEpoch(path)
    epochs_left = num_epoch - curr_epoch

    # Load Model
    if curr_epoch is not -1:
        model.load_state_dict(torch.load(path + '/' + str(curr_epoch)))
        # print(model)
        model.eval()

    for epoch in range(epochs_left):  # loop over the dataset multiple times
        running_loss = 0.0

        for data in train_dataloader:
            data = data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        loss_dict[model.name].append(running_loss)

        if epoch % 50 == 0:  # print every 50 mini-batches
            plotLoss(loss_dict, model.name)
            plotPointCloud(obj, model)
            saveModel(path, model, epoch)
            print(epoch)
