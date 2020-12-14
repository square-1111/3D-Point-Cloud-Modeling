import os, torch
import torch.autograd as autograd

from utils import currEpoch, saveModel, plotLoss, plotPointCloud, noiseFunc

    

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """
    Calculates the gradient penalty loss for WGAN GP

    Args:
    D             : discriminator network
    real_samples  : Real data samples
    fake_samples  : Fake data samples
    device        : GPU or CPU device
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.shape[0], 1, 1)
    alpha = alpha.expand_as(real_samples).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.to(device).requires_grad_(True)

    d_interpolates = D(interpolates)

    fake = torch.ones(real_samples.shape[0], 1).to(device).requires_grad_(False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

def trainingLoopGAN(obj, training_generator, generator, discriminator,
                    model_name, num_epoch, optimizer_g, optimizer_d, 
                    loss_dict, path, device, ae=None, mu=0, sigma=0.2, 
                    discriminator_boost=5, lambda_gp = 10):
    """
    A universal training loop to optimize any loss function

    Args:

    obj       : Object on which model is to be trained
    training_generator  : Training Set Generator
    generator           : Architecture of the generator
    discriminator       : Architecture of the generator
    num_epoch           : number of iterations
    optimizer_g         : optimize the loss for generator using this optimizer
    optimizer_d         : optimize the loss for discriminator using this optimizer
    loss_dict           : A dictionary to keep track of loss
    path                : location where to store model
    device              : GPU or CPU device
    ae                  : Autoencoder Model
    mu, sigma           : Mean and Standard Deviation for Normal Distribution 
    discriminator_boost : For every training iteration of generator train the 
                            critic this many times
    lambda_gp           : Regularizing factor for Gradient Penalty
    """

    # Check if path to directory exist. If no: then create one
    if not os.path.exists(path):
        os.makedirs(path)

    
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss_g = 0.0
        running_loss_d = 0.0

        for data in training_generator:
            data = data.to(device)
            if ae is not None:
                data = ae(data)
            for _ in range(discriminator_boost):
                # zero the parameter gradients
                optimizer_d.zero_grad()
                
                # forward + backward + optimize
                noise = noiseFunc(mu, sigma, data.shape[0], device)
                outputs = generator(noise)
                loss_d_fake = discriminator(outputs).mean()
                loss_d_fake.backward()

                loss_d_real = discriminator(data).mean()
                loss_d = loss_d_real + loss_d_fake
                
                # Gradient Penalty for Latent GAN
                if 'Latent' in generator.name:
                    grad_penal = compute_gradient_penalty(discriminator, data, outputs, device)
                    loss_d = loss_d + lambda_gp * grad_penal
                optimizer_d.step()
                running_loss_d += loss_d

            optimizer_g.zero_grad()
            noise = torch.randn((50, 128)).to(device)
            outputs = generator(noise)
            loss_g = discriminator(outputs).mean()
            loss_g.backward()
            optimizer_g.step()
            running_loss_g += loss_g

        loss_dict[generator.name].append(running_loss_g)  
        loss_dict[discriminator.name].append(running_loss_d)

        if epoch % 50 == 0:    # print every 10 mini-batches
            plotLoss(loss_dict, model_name)
            plotPointCloud(obj, generator)
            saveModel(path + 'Gen ', generator, epoch)
            saveModel(path + 'Dis ', discriminator, epoch)
            print(epoch)