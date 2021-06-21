import numpy as np
import MetropolisSampling as ms
import minresQLP2 as mqlp
import StochasticReconfiguration
import recenter as rc
import random
from copy import deepcopy
import jax.numpy as jnp

### Parameters
gamma = 0.01
n_vis = 5
n_hid = 7

### Initialize weights & biases
weights = np.array([[complex(random.uniform(0,1)*1e-3,random.uniform(0,1)*1e-3) for i in range(n_vis)] for j in range(n_hid)])
visBias = np.array([complex(random.uniform(0,1)*1e-3,random.uniform(0,1)*1e-3) for i in range(n_vis)])
#visBias = np.array([complex(0.,0.) for i in range(n_vis)])
hidBias = np.array([complex(random.uniform(0,1)*1e-3,random.uniform(0,1)*1e-3) for i in range(n_hid)])
#hidBias = np.array([complex(0.,0.) for i in range(n_hid)])
totParams = len(visBias)*len(hidBias) + len(visBias) + len(hidBias)

### Flatten parameters for constructing X
updatedParams = np.concatenate((weights.flatten(),visBias.flatten(),hidBias.flatten()))
updatedParams = np.reshape(updatedParams,totParams)
updatedParams2 = deepcopy(updatedParams)

spins = np.array([-1.,1.,-1.,1.,1.])

### Initialize training loop
for i in range(100):
  gamma = np.array([np.complex(.001,.001) for ii in range(totParams)])

  ### SamplingData - OFull, OAvg, EFull, EAvg, spins
  ### Ns: Number of samples to take
  Ns = 500

  ### Runs Metropolis Hastings (MetropolisSampling.py -> StochasticReconfiguration.py)
  OFull,OAvg,EAvg,EFull,spins = ms.MetropolisHastings(1000, Ns, weights, visBias, hidBias, spins)
  xCenter, eCenter = rc.recenter(OAvg,OFull,EAvg,EFull,Ns,totParams)

  ### Calculates Force and Covariance (I'm using the full Covariance with MinRes for testing)
  F, S = rc.ForceVec(xCenter,eCenter)

  ### The minres input functions
  #XFunReal = lambda x: ([np.matmul(np.conj(xCenter.real.T),np.matmul(xCenter.real,x))])
  #XFunImag = lambda x: ([np.matmul(np.conj(xCenter.imag.T),np.matmul(xCenter.imag,x))])
  #XFunReal = lambda x: ([np.conj(xCenter.real.T) @ xCenter.real @ x])
  #XFunImag = lambda x: ([np.conj(xCenter.imag.T) @ xCenter.imag @ x])

  ### Implement MinRes
  Nu = mqlp.MinresQLP(np.array([S]),F,1e-6,100)
  Nu = Nu[0]
  Nu = Nu.flatten()
  #print(Nu)
  #NuReal = mqlp.MinresQLP(XFunReal,F.real,1e-6,100)
  #NuReal = NuReal[0]

  #NuImag = mqlp.MinresQLP(np.array([S.imag]),F.imag,1e-6,100)
  #NuImag = mqlp.MinresQLP(XFunImag,F.imag,1e-6,100)
  #NuImag = NuImag[0]
  #print(NuImag)
  #for i in range(605):
  #NuImag = NuImag.flatten()*1.j
  #NuReal = NuReal.flatten()
  #print(NuReal)
  #print(NuImag)

  print("Iteration:", i)
  #print(jnp.multiply(gamma.imag, NuImag))

  ### Update parameters
  #RR = updatedParams.real
  #II = updatedParams.imag

  #RR -= jnp.multiply(gamma.real,NuReal)
  #II -= jnp.multiply(gamma.imag, NuImag)
  updatedParams -= jnp.multiply(gamma,Nu) 
  #print(updatedParams)
  weights = np.array([updatedParams[iii] for iii in range(n_vis*n_hid)])
  weights = weights.reshape((n_hid,n_vis))
  visBias = np.array([updatedParams[iii + n_vis*n_hid] for iii in range(n_vis)])
  hidBias = np.array([updatedParams[iii + n_vis*n_hid + n_vis] for iii in range(n_hid)])
