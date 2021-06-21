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
	### Ratio Function and Updates from Metropolis Sampling ###

	### theta = b_j + sum W_ij * v_j
	spinUpdate = deepcopy(spins)

	preFact = np.exp(-2*visBias[spinSite]*spins[spinSite])
	theta = SR.thetaCalc(spinUpdate, weights, hidBias)
	numer = jnp.prod(jnp.cosh(theta - 2 * weights[:,spinSite]*spins[spinSite]))
	denom = jnp.prod(jnp.cosh(theta))
	multFact = preFact*(jnp.divide(numer,denom))

	r = random.uniform(0,1)
	if np.argmin([1., abs(multFact)**2]) >= r:
		if spinUpdate[spinSite] == 1.:
			spinUpdate[spinSite] = -1.
		else:
			spinUpdate[spinSite] = 1.
	return spinUpdate, multFact

def ratioFunk_np(weights, visBias, hidBias, spins, spinSite):
	spinUpdate = deepcopy(spins)

	preFact = np.exp(-2*visBias[spinSite]*spins[spinSite])
	theta = SR.thetaCalc(spinUpdate, weights, hidBias)
	numer = complex(0.,0.)
	denom = complex(0.,0.)

def MetropolisHastings(steps, sampling, weights, visBias, hidBias, spins):
	''' Here we calculate the Metropolis step and sample ELoc, Expected Energy
	    and O Derivatives from these steps'''

	totParam = len(visBias)*len(hidBias) + len(visBias) + len(hidBias)

	OFull = np.array([[complex(0.,0.) for i in range(totParam)] for j in range(steps-sampling)])
	O = np.array([complex(0.,0.) for i in range(totParam)])
	EFull = np.array([complex(0.,0.) for i in range(steps-sampling)])
	EAvg = complex(0.,0.)
	count = 0.
	ExpectedEnergy = complex(0.,0.)
	updatedSpins = deepcopy(spins)
	for i in range(steps):
		spinSite = randint(0,len(visBias)-1)
		updatedSpins2 = ratioFunk(weights, visBias, hidBias, updatedSpins, spinSite)
		updatedSpins = updatedSpins2[0]

		if i%2 == 0:
			count += 1.

			OFull[i-sampling] = SR.O_Deriv_np(updatedSpins,weights, visBias, hidBias)
			O += OFull[i-sampling] 

			ELoc = SR.LocalEnergy_np(updatedSpins,weights,visBias, hidBias)
			EAvg -= ELoc

			EFull[i-sampling] = -ELoc

			ExpectedEnergy += HamiltonianExpectation(1, 1, weights, visBias, hidBias, updatedSpins)
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
	### Sampling the Expected Energy 

	Energy = 0.
	for i in range(5):
		theta = SR.thetaCalc(spins, weights, hidBias)
		WSpin = 2 * weights[:,i]*spins[i]
		numer = jnp.cosh(theta - WSpin)
		denom = jnp.cosh(theta)
		term1 = jnp.exp(-2*spins[i]*visBias[i])*jnp.prod(numer)/jnp.prod(denom)
		term2 = spins[i]*spins[(i+1)%5]
		#print(term2)
		Energy -= A*term1
		Energy -= B*term2
	#print(Energy)



	'''for i in range(len(spins)):
					operand = ratioFunk(weights, visBias, hidBias, spins, i)
					Energy += -A * operand[1]
					EnergyB = -B * spins[i] * spins[(i+1)%len(spins)]
					print(EnergyB)
					Energy+= EnergyB'''
	return Energy