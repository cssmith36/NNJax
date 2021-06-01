import numpy as np
import MetropolisSampling as ms
import minresQLP2 as mqlp
import StochasticReconfiguration
import recenter as rc
import random


gamma = 0.01
n_vis = 5
n_hid = 100

weights = np.array([[complex(random.uniform(0,1)*1e-2,random.uniform(0,1)*1e-2) for i in range(n_vis)] for j in range(n_hid)])
visBias = np.array([complex(random.uniform(0,1)*1e-2,random.uniform(0,1)*1e-2) for i in range(n_vis)])
hidBias = np.array([complex(random.uniform(0,1)*1e-2,random.uniform(0,1)*1e-2) for i in range(n_hid)])
totParams = len(visBias)*len(hidBias) + len(visBias) + len(hidBias)

updatedParams = np.concatenate((weights.flatten(),visBias.flatten(),hidBias.flatten()))
updatedParams = np.reshape(updatedParams,totParams)

spins = np.array([-1.,1.,-1.,1.,1.])
for i in range(100):
  count = 0
  gamma = 0.1

  ### SamplingData - OFull, OAvg, EFull, EAvg, spins
  Ns = 500
  OFull,OAvg,EAvg,EFull,spins = ms.MetropolisHastings(1000, Ns, weights, visBias, hidBias, spins)
  xCenter, eCenter = rc.recenter(OAvg,OFull,EAvg,EFull,Ns,totParams)
  F = rc.ForceVec(xCenter,eCenter)

  XFunReal = lambda x: ([np.matmul(np.conj(xCenter.real).T,np.matmul(xCenter.real,x))])
  XFunImag = lambda x: ([np.matmul(np.conj(xCenter.imag).T,np.matmul(xCenter.imag,x))])
  NuReal = mqlp.MinresQLP(XFunReal,F.real,1e-6,100)
  NuReal = NuReal[0]
  NuImag = mqlp.MinresQLP(XFunImag,F.imag,1e-6,100)
  NuImag = NuImag[0]
  print("Here!")
  print(len(NuReal))
  for j in range(totParams):
    updatedParams[j] = updatedParams[j] - gamma*(NuReal[j][0] + NuImag[j][0])
    count += 1