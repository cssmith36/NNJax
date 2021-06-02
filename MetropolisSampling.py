import numpy as np
import jax.numpy as jnp
import networkx as nx
import StochasticReconfiguration as SR
from copy import deepcopy
from random import randint
import random

def HilbertBuild(edges):
	return edges

def ratioFunk(weights, visBias, hidBias, spins, spinSite):

	### theta = b_j + sum W_ij * v_j
	spinUpdate = deepcopy(spins)

	preFact = jnp.exp(-2*visBias[spinSite]*spins[spinSite])
	theta = SR.thetaCalc(spinUpdate, weights, hidBias)
	numer = jnp.prod(jnp.cosh(theta - 2 * weights[:,spinSite]*spins[spinSite]))
	denom = jnp.prod(jnp.cosh(theta))
	multFact = preFact*(numer/denom)
	#print(multFact)
	r = random.uniform(0,1)
	if np.argmin([1., abs(multFact)**2]) >= r:
		#print("flip")
		if spinUpdate[spinSite] == 1.:
			spinUpdate[spinSite] = -1.
		else:
			spinUpdate[spinSite] = 1.
	return spinUpdate, multFact

def MetropolisHastings(steps, sampling, weights, visBias, hidBias, spins):
	''' Hilbert: Graph with the given connections'''

	totParam = len(visBias)*len(hidBias) + len(visBias) + len(hidBias)

	OFull = np.array([[complex(0.,0.) for i in range(totParam)] for j in range(steps-sampling)])
	O = np.array([complex(0.,0.) for i in range(totParam)])
	print("O:",len(O))
	EFull = np.array([complex(0.,0.) for i in range(steps-sampling)])
	EAvg = complex(0.,0.)
	count = 0.
	updatedSpins = deepcopy(spins)
	for i in range(steps):
		spinSite = randint(0,len(visBias)-1)
		updatedSpins = ratioFunk(weights, visBias, hidBias, updatedSpins, spinSite)
		updatedSpins = updatedSpins[0]

		ExpectedEnergy = 0.

		if i >= sampling:
			count += 1.

			O += SR.O_Deriv(updatedSpins,weights, visBias, hidBias)
			OFull[i-sampling] = O

			ELoc = SR.LocalEnergy(spins,updatedSpins,weights, visBias, hidBias)
			EAvg += ELoc

			EFull[i-sampling] = ELoc

			ExpectedEnergy += HamiltonianExpectation(1, 1, weights, visBias, hidBias, spins)
	ExpectedEnergy = ExpectedEnergy/count
	print(spins)
	print(updatedSpins)
	print(ExpectedEnergy)

	OAvg = O/count
	EAvg = EAvg/count

	print(EAvg)
	#print(OAvg)
	return OFull, OAvg, EAvg, EFull, updatedSpins

def HamiltonianExpectation(A, B, weights, visBias, hidBias, spins):
	Energy = 0.
	for i in range(len(spins)):
		operand = ratioFunk(weights, visBias, hidBias, spins, i)
		Energy += -A * operand[1]
		Energy += -B * spins[i] * spins[(i+1)%len(spins)]
	return Energy