from SBML.PyTorchZonotope import core
import torch
import math
import time

def train(model,loss,optimizer,xTrain,yTrain,epochs,batchsize=64,noise=0.0,verbose=True) -> torch.nn.Module:
    """
    train: NN-trainig (set and point-based)
    =======================================

    This function returns a trained PyTorch model for given input data and targets. 
    
    Parameters:
    -----------
    - model: Neural network torch model
    - loss: loss function for nn training 
    - optimizer: inintialized optimizer with weights of the model
    - xTrain: torch tensor xData with size (num. samples, input size)
    - yTrain: torch tensor yData with size (num. samples, output size)
    - epochs: number of training epochs
    - batchsize: mini batchsize for training
    - noise: perturbation radius for set-based trainig 
    - verbose: boolean for printing learning parameters and learning history
    """
    if verbose:
        printPartameters(['Epochs','Batchsize','Perturbation Radius'],[epochs,batchsize,noise])

    numSamples = xTrain.shape[0]
    numItter = int(math.floor(numSamples/batchsize))

    startTime = time.time()

    for epoch in range(epochs):
        indxShuffled = torch.randperm(xTrain.shape[0])
        epochLoss = 0
        for itter in range(numItter):
            xBatch = xTrain[indxShuffled[itter*batchsize:(itter+1)*batchsize],...]

            optimizer.zero_grad()
            if noise > 0:
                xBatch = xBatch.t()
                outputs = model(core.Zonotope(torch.cat([xBatch.unsqueeze(1),(noise*torch.eye(xBatch.size(0),device=xBatch.device).unsqueeze(2)).repeat(1,1,batchsize)],dim=1)))
                yBatch = core.Zonotope(yTrain[indxShuffled[itter*batchsize:(itter+1)*batchsize],...].t())
            else:
                outputs = model(xBatch)
                yBatch = yTrain[indxShuffled[itter*batchsize:(itter+1)*batchsize],...]

            lossVal = loss(outputs,yBatch)
            lossVal.backward()
            optimizer.step()
        
        epochLoss += lossVal
        epochLoss /= numItter
        epochTime = time.time()
        if verbose:
            print('|',epoch,'\t|{0:.1}'.format(epochTime-startTime),'\t|{0:.1}'.format(epochLoss.item()),'\t|')

    return model

def printPartameters(params, values):
    """Prints the Neural Network Learning Paramteres and the learning logging header"""
    print('_____________________________________')
    print(' Neural Network Training Paramteres: ')
    print('-------------------------------------')
    print('Training Parameters:')
    for i in range(len(params)):
        print('\t',params[i],' ',values[i])
    print('______________________________________')
    print('|Epoch\t|Time\t|Loss')
    print('--------------------------------------')