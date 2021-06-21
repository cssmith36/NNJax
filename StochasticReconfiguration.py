import numpy as np
import jax.numpy as jnp
import cmath


### Theta Calculation
def thetaCalc(spins, weights, hidBias):
	theta = hidBias + jnp.dot(weights,spins)
	'''theta = np.array([complex(0.,0.) for i in range(len(hidBias))])
				for i in range(len(hidBias)):
					for j in range(len(spins)):
						theta[i] += weights[i,j]*spins[j]
			'''
	return theta


### Local Energy Calculation
def LocalEnergy(spins, weights, visBias, hidBias):

	ELoc = np.array([complex(0.,0.) for i in range(len(spins))])
	shiftedSpins = np.array([spins[(i+1)%5] for i in range(5)])
	weights2 = weights.T

	### 1st ELoc contribution
	ELocJ = jnp.multiply(spins,shiftedSpins)
	ELocJ = complex(1.,0.)
	#print(jnp.sum(ELocJ))

	theta = thetaCalc(spins,weights,hidBias)
	preFact = jnp.exp(-2*jnp.multiply(visBias,spins))
	#preFact = 1.
	#print(preFact)
	for i in range(len(spins)):
		multArray = jnp.divide(jnp.cosh(theta - 2 * weights2[i]*spins[i]),jnp.cosh(theta))
		### 2nd & 3rd ELoc contributions
		#print("Here")
		#print(preFact[i]*jnp.prod(multArray))
		ELoc[i] += preFact[i]*jnp.prod(multArray)
		#print(jnp.prod(multArray))
		#ELoc[i] += 1j*(-1)**((1+spins[i])/2)*jnp.prod(multArray) * preFact[i]
	#print(ELoc)
	#print(multArray)
	ELoc = jnp.sum(ELoc) + ELocJ
	#print(ELoc)
	return ELoc

def LocalEnergy_np(spins, weights, visBias, hidBias):
	#ELoc = np.array([complex(0.,0.) for i in range(len(spins))])
	weights2 = weights.T
	ELocJ = complex(0.,0.)
	theta = thetaCalc(spins, weights, hidBias)

	preFactReal = np.array([complex(0.,0.) for i in range(5)])
	preFactImag = np.array([complex(0.,0.) for i in range(5)])
	multArrayReal = np.array([complex(1.,0.) for i in range(5)])
	multArrayImag = np.array([complex(0.,1.) for i in range(5)])

	multProdReal = complex(1.,0.)
	multProdImag = complex(0.,1.)

	for i in range(len(spins)):
		ELocJ += spins[i]*spins[(i+1)%5]
		preFactReal[i] = np.exp(-2*visBias[i].real*spins[i])
		preFactImag[i] = np.exp(-2*visBias[i].imag*spins[i])
		for j in range(len(hidBias)):
			numerReal = np.cosh(theta[i].real - weights2[i][j].real*spins[i])
			denomReal = np.cosh(theta[i].real)
			multArrayReal[i] *= numerReal/denomReal

			numerImag = np.cosh(theta[i].imag - weights2[i][j].imag*spins[i])
			denomImag = np.cosh(theta[i].imag)
			multArrayImag[i] *= numerImag/denomImag
		multProdReal *= preFactReal[i] * multArrayReal[i]
		multProdImag *= preFactImag[i] * multArrayImag[i]*1j

	ELoc = ELocJ + multProdReal + multProdImag
	return ELoc


def O_Deriv_np(spins, weights, visBias, hidBias):
	numHid = len(hidBias)
	numVis = len(visBias)
	O_a = spins

	theta = thetaCalc(spins, weights, hidBias)
	O_b = np.array([complex(0.,0.) for i in range(numHid)])

	for i in range(numHid):
		O_b[i] = np.tanh(theta[i])
	O_WReal = np.array([[complex(1.,0.) for i in range(numVis)] for j in range(numHid)])
	O_WImag = np.array([[complex(0.,1.) for i in range(numVis)] for j in range(numHid)])
	for i in range(numHid):
		for j in range(numVis):
			O_WReal[i][j] = O_b[i].real * spins[j]
			O_WImag[i][j] = O_b[i].imag * spins[j]
	O_W = O_WReal + O_WImag
	O_W = O_W.flatten()

	StackDev = np.concatenate([O_W,O_a,O_b])
	StackDev = StackDev.flatten()
	return StackDev


def O_Deriv(spins, weights, visBias, hidBias):

	numHid = len(hidBias)
	numVis = len(visBias)
	#print(numVis)
	### Calculate derivatives
	O_a = spins
	theta = thetaCalc(spins, weights, hidBias)
	O_b = jnp.tanh(theta)
	#print(O_b)
	#print(spins)
	'''O_W = np.array([[np.complex(0.,0.) for i in range(numVis)] for j in range(numHid)])
	for i in range(numHid):
		for j in range(numVis):
			O_W[i][j] = O_b[i]*np.complex(spins[j],0.)'''

	O_W = jnp.kron(spins, O_b)
	#O_W2 = jnp.kron(O_b,spins)
	#O_W = O_W.flatten()
	#O_W2 = np.reshape(O_W2,(numHid,numVis))
	#print(O_W-O_W2)
	#print(np.shape(O_W))
	StackDev = np.concatenate([O_W,O_a,O_b])
	StackDev = StackDev.flatten()
	#print(np.shape(StackDev))

	return StackDev


