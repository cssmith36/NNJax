import numpy as np
#import jax.numpy as jnp
import cmath


def LocalEnergy(spins, updatedSpins, weights, visBias, hidBias):

	ELoc = np.array([complex(0.,0.) for i in range(len(spins))])

	for k in range(5):
		ELoc[k] += spins[k] * updatedSpins[k]

		preFact = np.exp(-2*visBias[k]*spins[k])
		multFact = 1
		for i in range(len(hidBias)):
			### Construct theta
			theta = 0.
			for j in range(len(visBias)):
				theta += weights[i][j] * spins[j]
			theta += hidBias[i]
			multFact *= np.cosh(theta - 2 * weights[i][k])/np.cosh(theta)
		multFact *= preFact
		multFact2 = (-1)**((1+spins[k])/2)*complex(0,1)*multFact

		ELoc[k] += multFact
		ELoc[k] += multFact2
	ELoc = np.sum(ELoc)
	#print(ELoc)
	return ELoc

def O_Deriv(spins, weights, visBias, hidBias):

	numHid = len(hidBias)
	numVis = len(visBias)

	O_a = np.array([complex(0.,0.) for i in range(len(spins))])
	O_b = np.array([complex(0.,0.) for i in range(len(hidBias))])
	O_W = np.array([[complex(0.,0.) for i in range(numVis)] for j in range(numHid)])
	#print(O_W)
	for i in range(len(spins)):
		O_a[i] = spins[i]
	for i in range(len(hidBias)):
		theta = 0.
		for j in range(len(visBias)):
			theta += weights[i][j] * spins[j]
		theta += hidBias[i]

		O_b[i] = np.tanh(theta)

		for j in range(numVis):
			#print("i,j:",i,j)
			O_W[i][j] = spins[j] * np.tanh(theta)


	O_W = O_W.flatten()
	StackDev = np.concatenate([O_W,O_a,O_b])
	StackDev = StackDev.flatten()

	return StackDev

