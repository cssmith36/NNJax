import numpy as np
import jax.numpy as jnp
import cmath


def thetaCalc(spins, weights, hidBias):
	theta = hidBias + jnp.dot(weights,spins)
	#print(theta)
	return theta

def LocalEnergy(spins, updatedSpins, weights, visBias, hidBias):

	ELoc = np.array([complex(0.,0.) for i in range(len(spins))])

	ELocJ = jnp.sum(jnp.dot(spins,updatedSpins))

	#print(type(ELoc))

	theta = thetaCalc(spins,weights,hidBias)
	preFact = jnp.exp(-2*jnp.multiply(visBias,spins))
	for i in range(len(spins)):
		multArray = jnp.cosh(theta - 2 * weights[:,i]*spins[i])/jnp.cosh(theta)
		#print(multArray)
		#print(preFact)
		ELoc[i] += preFact[i]*jnp.prod(multArray)
		ELoc[i] += 1j*(-1)**((1+spins[i])/2)
	ELoc = jnp.sum(ELoc) + ELocJ
	#print(ELoc)
	return ELoc

def O_Deriv(spins, weights, visBias, hidBias):

	numHid = len(hidBias)
	numVis = len(visBias)

	#print(O_W)
	O_a = spins
	theta = thetaCalc(spins, weights, hidBias)
	O_b = jnp.tanh(theta)
	#print(O_b)
	O_W = jnp.kron(O_b,spins)
	#print(O_W)

	StackDev = np.concatenate([O_W,O_a,O_b])
	StackDev = StackDev.flatten()
	#print(StackDev)

	return StackDev

