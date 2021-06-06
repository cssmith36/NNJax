import numpy as np
import jax.numpy as jnp
import cmath


### Theta Calculation
def thetaCalc(spins, weights, hidBias):
	theta = hidBias + jnp.dot(weights,spins)
	return theta

### Local Energy Calculation
def LocalEnergy(spins, weights, visBias, hidBias):

	ELoc = np.array([complex(0.,0.) for i in range(len(spins))])
	shiftedSpins = np.array([spins[(i+1)%5] for i in range(5)])

	### 1st ELoc contribution
	ELocJ = jnp.sum(jnp.dot(spins,shiftedSpins))

	theta = thetaCalc(spins,weights,hidBias)
	preFact = jnp.exp(-2*jnp.multiply(visBias,spins))
	for i in range(len(spins)):
		multArray = jnp.cosh(theta - 2 * weights[:,i]*spins[i])/jnp.cosh(theta)
		### 2nd & 3rd ELoc contributions
		ELoc[i] += preFact[i]*jnp.prod(multArray)
		ELoc[i] += 1j*(-1)**((1+spins[i])/2)*jnp.prod(multArray)
	ELoc = jnp.sum(ELoc) + ELocJ
	return ELoc

def O_Deriv(spins, weights, visBias, hidBias):

	numHid = len(hidBias)
	numVis = len(visBias)
	### Calculate derivatives
	O_a = spins
	theta = thetaCalc(spins, weights, hidBias)
	O_b = jnp.tanh(theta)
	O_W = jnp.kron(O_b,spins)

	StackDev = np.concatenate([O_W,O_a,O_b])
	StackDev = StackDev.flatten()

	return StackDev

