import numpy as np

def recenter(OAvg,OFull,EAvg,EFull,Ns,totParams):
	xCenter = np.array([[complex(0., 0.) for i in range(totParams)] for j in range(Ns)])
	eCenter = np.array([complex(0., 0.) for i in range(Ns)])
	for i in range(Ns):
		xCenter[i] = (1/np.sqrt(Ns-1)*(OFull[i]-OAvg))
		eCenter[i] = (1/np.sqrt(Ns-1)*(EFull[i]-EAvg))
	return xCenter, eCenter

def ForceVec(xCenter, eCenter):
	F = np.matmul(np.conj(xCenter).T,eCenter)
	return F