from SBRL.PyTorchZonotopeExtension import core
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
    - xTrain: xData with size (input size, num. samples)
    - yTrain: yData with size (output sizem num. samples)
    - epochs: number of training epochs
    - batchsize: mini batchsize for training
    - noise: perturbation radius for set-based trainig 
    - verbose: boolean for printing learning parameters and learning history
    """
    if verbose:
        printPartameters(['Epochs','Batchsize','Perturbation Radius'],[epochs,batchsize,noise])

    numSamples = xTrain.shape[1]
    numItter = int(math.floor(numSamples/batchsize))

    startTime = time.time()

    for epoch in range(epochs):
        indxShuffled = torch.randperm(xTrain.size(1))
        epochLoss = 0
        for itter in range(numItter):
            xBatch = torch.as_tensor(xTrain[:,indxShuffled[itter*batchsize:(itter+1)*batchsize]],dtype=torch.float32)

            optimizer.zero_grad()
            if noise > 0:
                outputs = model(core.Zonotope(torch.cat([xBatch.unsqueeze(1),(noise*torch.eye(xBatch.size(0)).unsqueeze(2)).repeat(1,1,batchsize)],dim=1)))
                yBatch = core.Zonotope(yTrain[:,indxShuffled[itter*batchsize:(itter+1)*batchsize]])
            else:
                outputs = model(xBatch.t())
                yBatch = torch.as_tensor(yTrain[:,indxShuffled[itter*batchsize:(itter+1)*batchsize]],dtype=torch.float32).t()

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