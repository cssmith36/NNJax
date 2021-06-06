import numpy as np
import MetropolisSampling as ms
import minresQLP2 as mqlp
import StochasticReconfiguration
import recenter as rc
import random

### Parameters
gamma = 0.01
n_vis = 5
n_hid = 100

### Initialize weights & biases
weights = np.array([[complex(random.uniform(0,1)*1e-2,random.uniform(0,1)*1e-2) for i in range(n_vis)] for j in range(n_hid)])
print("init")
print(np.shape(weights))
visBias = np.array([complex(random.uniform(0,1)*1e-2,random.uniform(0,1)*1e-2) for i in range(n_vis)])
hidBias = np.array([complex(random.uniform(0,1)*1e-2,random.uniform(0,1)*1e-2) for i in range(n_hid)])
totParams = len(visBias)*len(hidBias) + len(visBias) + len(hidBias)

### Flatten parameters for constructing X
updatedParams = np.concatenate((weights.flatten(),visBias.flatten(),hidBias.flatten()))
updatedParams = np.reshape(updatedParams,totParams)

spins = np.array([-1.,1.,-1.,1.,1.])

### Initialize training loop
for i in range(100):
  gamma = 0.1

  ### SamplingData - OFull, OAvg, EFull, EAvg, spins
  Ns = 500

  ### Runs Metropolis Hastings
  OFull,OAvg,EAvg,EFull,spins = ms.MetropolisHastings(1000, Ns, weights, visBias, hidBias, spins)
  xCenter, eCenter = rc.recenter(OAvg,OFull,EAvg,EFull,Ns,totParams)

  ### Calculates Force and Covariance (I'm using the full Covariance with MinRes for testing)
  F, S = rc.ForceVec(xCenter,eCenter)

  #XFunReal = lambda x: ([np.matmul(np.conj(xCenter.real.T),np.matmul(xCenter.real,x))])
  #XFunImag = lambda x: ([np.matmul(np.conj(xCenter.imag.T),np.matmul(xCenter.imag,x))])
  XFunReal = lambda x: ([np.conj(xCenter.real.T) @ xCenter.real @ x])
  XFunImag = lambda x: ([np.conj(xCenter.imag.T) @ xCenter.imag @ x])
  print(np.shape(S.real))

  ### Implement MinRes
  NuReal = mqlp.MinresQLP(np.array([S.real]),F.real,1e-6,100)
  #NuReal = mqlp.MinresQLP(XFunReal,F.real,1e-6,100)
  NuReal = NuReal[0]
  NuImag = mqlp.MinresQLP(np.array([S.imag]),F.imag,1e-6,100)
  #NuImag = mqlp.MinresQLP(XFunImag,F.imag,1e-6,100)
  NuImag = NuImag[0]
  print("Iteration:", i)

  ### Update parameters
  updatedParams = updatedParams - gamma*(NuReal[:][0] + NuImag[:][0])

  weights = np.array([updatedParams[i] for i in range(n_vis*n_hid)])
  weights = weights.reshape((n_hid,n_vis))
  visBias = np.array([updatedParams[i + n_vis*n_hid] for i in range(n_vis)])
  hidBias = np.array([updatedParams[i + n_vis*n_hid + n_vis] for i in range(n_hid)])
