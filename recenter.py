import numpy as np

def recenter(OAvg,OFull,EAvg,EFull,Ns,totParams):
	xCenter = np.array([[complex(0., 0.) for i in range(totParams)] for j in range(Ns)])
	eCenter = np.array([complex(0., 0.) for i in range(Ns)])
	for i in range(Ns):
		xCenter[i] = (1/np.sqrt(Ns-1)*(OFull[i]-OAvg))
		eCenter[i] = (1/np.sqrt(Ns-1)*(EFull[i]-EAvg))
	#print("Recentered:")
	#print(np.shape(xCenter))
	#print(EAvg)
	#print(eCenter)
	#print("Here")
	#print(xCenter)
	#print(eCenter)
	return xCenter, eCenter

def ForceVec(xCenter, eCenter):
	FReal = np.matmul(np.conj(xCenter.real).T,eCenter.real)
	FImag = np.matmul(np.conj(xCenter.imag).T,eCenter.imag)
	SImag = np.matmul(np.conj(xCenter.imag).T,xCenter.imag)
	SReal = np.matmul(np.conj(xCenter.real).T,xCenter.real)
	F = FReal + FImag
	S = SReal + SImag
	return F, S