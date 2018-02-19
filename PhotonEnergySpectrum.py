from wavFunc22 import *
import csv
from scipy.integrate import *

def getPhotonData(): 
	root=tk.Tk()
	root.withdraw()
	fil=filedialog.askopenfile()
	sfil=filedialog.asksaveasfile(mode='w',defaultextension='.txt')

	data=np.load(fil.name)
	acs=data['acs']
	ts=data['ts']
	photx, photy=getFFTPowerSpectrum(acs,(ts[-1]-ts[0])/len(ts))
	photy*=(4*np.pi*1e-7*1.6*1e-19)/(6*np.pi*3e8)

	harmnumber=[]
	freqs=[]
	energies=[]
	photnumber=[]
	f0=(3e8)/data['lam']
	firstharmenergy=4.13567e-15*f0

	for i in range(1,16):
		indexlist=get_indices(f0,i*f0,photx) 
		harmnumber.append(i)
		freqs.append(i*f0)
		energies.append(simps(y=photy[indexlist],x=photx[indexlist]))
		photnumber.append(energies[i-1]/(i*firstharmenergy))

	with open(sfil.name,'w') as f:
		writer=csv.writer(f,delimiter='\t')
		writer.writerows(zip(harmnumber,freqs,energies,photnumber))

def plotPhotonData(): 
	root=tk.Tk()
	root.withdraw()
	fil=filedialog.askopenfile()
	photdata=np.loadtxt(fil.name).T
	fig,ax=plt.subplots(1,1)
	ax.bar(photdata[0],photdata[3])
	ax.set_xlabel("Harmonic Multiple")
	ax.set_ylabel("Photon Number")
	ax.set_title("Radiated Photon Number for %sGeV/m field and %sfs std"%(input("Emax: "),input("std: ")))
	plt.show()

def get_indices(fh,center,array): 
	width=0.5*fh
	halfinterval=int(width*1e-13)
	endindex=np.where(array>=center)[0][0]+halfinterval
	startindex=np.where(array>=center)[0][0]-halfinterval
	return np.arange(startindex,endindex,1)

getPhotonData()
plotPhotonData()