import numpy as np 
import matplotlib.pyplot as plt

def getL(l,nx):
    L = np.zeros((nx,nx))
    for i in range(0,nx):
        for j in range(0,nx):
            dist = np.abs(i-j)
            if dist >= nx-l:
                dist = nx-dist
            L[i,j] = np.exp(-1.5*((dist/l)**2))
    
    # fig,ax = plt.subplots()
    # pos = ax.imshow(L,cmap='bwr',vmin=-1,vmax=1)
    # im_ratio = L.shape[0]/(L.shape[1])
    # cbar = fig.colorbar(pos,ax=ax,fraction=0.047*im_ratio)
    # ax.set_title('Localization Matrix')
    # ax.set_ylabel('X index')
    # ax.set_xlabel('X index')
    # plt.savefig('Homework6/LocalCov.png')
    # plt.show()
    # plt.close()
    return L 
