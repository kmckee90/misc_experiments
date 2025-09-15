#%%
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import clear_output

def WTA(vec, k=1, sample=True):
    prob = torch.Tensor.softmax(vec.squeeze(), 0)
    if sample:
        act = torch.zeros_like(prob)
        act[torch.Tensor.multinomial(prob.T, k)] = 1
        return act
    else:
        return prob

def RandomDoublyStochasticMatrices(dim1, dim2, iters=10):
    W = torch.rand(dim1, dim2, dim2)
    for iter in range(iters):
        Wsums = torch.sum(W, 1, keepdim=True) * torch.ones(1, dim2, 1)
        W = (W / Wsums)
        Wsums = torch.sum(W, 2, keepdim=True) * torch.ones(1, 1, dim2)
        W = (W / Wsums)
    return W

class RingWTAModel:
    def __init__(self, Wf, Wb,  nSteps=10, nStates=2, temp=1, coolRate=1, sampling = False):
        self.Wf = Wf
        self.Wb = Wb
        self.nSteps = nSteps
        self.nStates = nStates
        self.sampling = sampling
        self.temp = temp 
        self.coolRate = coolRate
        
        self.Z = torch.zeros(1, nSteps, nStates)
        self.Z[0,0,0]=1
        self.E = 0

    def step(self, input):
        Zn = torch.zeros_like(self.Z)
        self.E = 0
        for l in range(0,self.nSteps-1):
            Zn[:,l,:] = WTA( (self.Z[:,l-1, :] @ self.Wf[l-1,:,:] + self.Z[:,l+1, :] @ self.Wb[l,:,:] + input[l,:])/self.temp, 1, sample=self.sampling)
            m = torch.argmax(Zn[:,l,:])
            Zn[:,l,:] = torch.zeros_like(Zn[:,l,:])
            Zn[:,l,m]=1

        Zn[:,-1,:] = WTA( (self.Z[:,-2, :] @ self.Wf[-2,:,:] + self.Z[:,0, :] @ self.Wb[-1,:,:] + input[-1,:])/self.temp, 1, sample=self.sampling)
        m = torch.argmax(Zn[:,-1,:])
        Zn[:,-1,:] = torch.zeros_like(Zn[:,-1,:])
        Zn[:,-1,m]=1

        for l in range(0,self.nSteps-1):
            self.E -= torch.sum( (Zn[:,l-1,:].T @ Zn[:,l,:])*self.Wf[l-1,:,:] + Zn[:,l+1,:].T @ Zn[:,l,:]*self.Wb[l,:,:])        
        self.E -= torch.sum( (Zn[:,-2,:].T @ Zn[:,-1,:])*self.Wf[-2,:,:] + Zn[:,0,:].T @ Zn[:,-1,:]*self.Wb[-1,:,:])        

        self.Z = Zn.clone()
        self.temp *= self.coolRate

        return self.Z, self.E, self.temp

class MRFModel:
    def __init__(self, W, nSteps=10, temp=1, coolRate=1, sampling = False):
        self.W = W
        self.nSteps = nSteps
        self.sampling = sampling
        self.temp = temp 
        self.coolRate = coolRate
        self.Z = torch.zeros(1,nSteps)
        self.Z[0,0]=1
        self.E = 0

    def step(self, input):
        self.E = 0
        Zp = torch.sigmoid( (self.Z @ self.W + input) / self.temp)
        Zn = torch.Tensor.bernoulli(Zp)
        self.Z = Zn.clone()
        self.temp *= self.coolRate
        self.E -= torch.sum( (Zn.T @ Zn)*self.W ) 
        return self.Z, self.E, self.temp

def plotSim(iState1, eState1, tState1):
    figure = plt.figure(figsize=(10, 7), tight_layout=True)
    figure.add_subplot(3, 1, 1)
    plt.imshow(iState1.T, cmap="rainbow")
    figure.add_subplot(3, 1, 2)
    plt.plot(eState1)
    figure.add_subplot(3, 1, 3)
    plt.plot(tState1)
    plt.show()


class RBMModel:
    def __init__(self, Wf, Wb,  nSteps=10, nStates=2, temp=1, coolRate=1, sampling = False):
        self.Wf = Wf
        self.Wb = Wb
        self.nSteps = nSteps
        self.nStates = nStates
        self.sampling = sampling
        self.temp = temp 
        self.coolRate = coolRate
        
        self.Z = torch.zeros(1, nSteps, nStates)
        self.Z[0,0,0]=1
        self.E = 0

    def step(self, input):
        Zn = torch.zeros_like(self.Z)
        self.E = 0
        for l in range(1,self.nSteps-1):
            Zn[:,l,:] = WTA( (self.Z[:,l-1, :] @ self.Wf[l-1,:,:] + self.Z[:,l+1, :] @ self.Wb[l,:,:] + input[l,:])/self.temp, 1, sample=self.sampling)
            m = torch.argmax(Zn[:,l,:])
            Zn[:,l,:] = torch.zeros_like(Zn[:,l,:])
            Zn[:,l,m]=1


   #     Zn[:,0,:] = WTA( (self.Z[:,1, :] @ self.Wb[0,:,:]  + input[0,:])/self.temp, 1, sample=self.sampling)
   #     m = torch.argmax(Zn[:,0,:])
   #     Zn[:,0,:] = torch.zeros_like(Zn[:,0,:])
   #     Zn[:,0,m]=1

        Zn[:,-1,:] = WTA( (self.Z[:,-2, :] @ self.Wf[-2,:,:] + input[-1,:])/self.temp, 1, sample=self.sampling)
        m = torch.argmax(Zn[:,-1,:])
        Zn[:,-1,:] = torch.zeros_like(Zn[:,-1,:])
        Zn[:,-1,m]=1

        for l in range(1,self.nSteps-1):
            self.E -= torch.sum( (Zn[:,l-1,:].T @ Zn[:,l,:])*self.Wf[l-1,:,:] + Zn[:,l+1,:].T @ Zn[:,l,:]*self.Wb[l,:,:])        
        self.E -= torch.sum( (Zn[:,-2,:].T @ Zn[:,-1,:])*self.Wf[-2,:,:] )        
        self.E -= torch.sum( (Zn[:, 1,:].T @ Zn[:, 0,:])*self.Wb[0,:,:] )         

        self.Z = Zn.clone()
        self.temp *= self.coolRate

        return self.Z, self.E, self.temp




#%% Ring WTA network
nStates=4
nSteps=30
nFrame = 210
Wf = RandomDoublyStochasticMatrices(nSteps, nStates, 100)
Wb = Wf.T.permute((2,0,1))
#Wb = RandomDoublyStochasticMatrices(nSteps, nStates, 100)
M = RingWTAModel(Wf, Wb, sampling=True, nStates=nStates, nSteps=nSteps, temp=.01, coolRate=.95)
obsFrame = deque([], maxlen=nFrame)
energyFrame = deque([], maxlen=nFrame)
tempFrame = deque([], maxlen=nFrame)
input = torch.zeros(nSteps, nStates)

#%%
#M.temp=.5
M.coolRate = .975
M.sampling=True
input = torch.zeros(nSteps, nStates)

for t in range(200): #Run just a few less than nFrame to see the difference made.
    M.temp += 0.025*np.exp(-1/5**2*(t-30)**2)
    ob, energy, tempm = M.step(input)
    obsFrame.append(ob.squeeze().nonzero()[:,1].numpy() )
    energyFrame.append(energy)
    tempFrame.append(tempm)

obsHist = np.array(obsFrame)
plotSim(obsHist,  energyFrame, tempFrame)







#%% Fully connected random field of binary units.
nSteps=50
nFrame = 210
W = torch.randn(nSteps, nSteps)
W = (W+W.T)/2
M = MRFModel(W, sampling=True, nSteps=nSteps, temp=.01, coolRate=.95)
obsFrame = deque([], maxlen=nFrame)
energyFrame = deque([], maxlen=nFrame)
tempFrame = deque([], maxlen=nFrame)
input = torch.zeros(nSteps, nStates)

#%%
#M.temp = .01
M.coolRate = 0.975
M.sampling=True
input = torch.zeros_like(M.Z) 
#input[0,0:5]= -10
for t in range(200): #Run just a few less than nFrame to see the difference made.
    M.temp += 0.25*np.exp(-1/5**2*(t-30)**2)
    ob, energy, tempm = M.step(input)
    obsFrame.append(ob.squeeze().numpy())
    energyFrame.append(energy)
    tempFrame.append(tempm)

obsHist = np.array(obsFrame)
plotSim(obsHist,  energyFrame, tempFrame)

# %%






#%% Ring WTA network
nStates=2
nSteps=5
nFrame = 210
Wf = RandomDoublyStochasticMatrices(nSteps, nStates, 100)
Wb = Wf.T.permute((2,0,1))
#Wb = RandomDoublyStochasticMatrices(nSteps, nStates, 100)
M = RBMModel(Wf, Wb, sampling=True, nStates=nStates, nSteps=nSteps, temp=.01, coolRate=.95)
obsFrame = deque([], maxlen=nFrame)
energyFrame = deque([], maxlen=nFrame)
tempFrame = deque([], maxlen=nFrame)
input = torch.zeros(nSteps, nStates)

input[0,:] = [0,0,10,0,0]

#%%
#M.temp=.5
M.coolRate = .975
M.temp = .01
M.sampling=True
input = torch.zeros(nSteps, nStates)

for t in range(200): #Run just a few less than nFrame to see the difference made.
    M.temp += 0.025*np.exp(-1/5**2*(t-30)**2)
    ob, energy, tempm = M.step(input)
    obsFrame.append(ob.squeeze().nonzero()[:,1].numpy() )
    energyFrame.append(energy)
    tempFrame.append(tempm)

obsHist = np.array(obsFrame)
plotSim(obsHist,  energyFrame, tempFrame)



# %%
