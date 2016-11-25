# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 23:15:22 2016

@author: steve
"""


# TO TIME PORTIONS OF CODE:
#
# from timeit import default_timer as timer
# start = timer()
# end = timer()
# print(start-end)


import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso


np.random.seed(12345)


def generate_cfa(shape, channels=3):
    rows = shape[0]
    cols = shape[1]
    size = rows*cols
    
    beg_vec = np.zeros((size,1))
    cut_arr = np.sort(np.random.rand(size,channels-1))
    end_vec = np.ones((size,1))
    
    conc_arr = np.concatenate((beg_vec, cut_arr, end_vec), axis=1)
    diff_arr = np.diff(conc_arr);
    
    return np.reshape(diff_arr,(rows,cols,channels),'C')


def project_image_onto_cfa(image, cfa):
    projection = np.multiply(image, cfa)
    return np.sum(projection,axis=2)


def generate_dct(N):
    D = np.zeros((N, N))
    
    for k in range(N):
        for n in range(N):
            D[k,n] = np.cos((np.pi/N)*(n+0.5)*k)
    
    D[0,:] = D[0,:]*(1/np.sqrt(2))
    D = D*np.sqrt(2/N)
    
    return D


def generate_thetaetf():
    l = (1+np.sqrt(5))/2
    thetaetf = 1/np.sqrt(1+l**2)*np.array([[0,0,1,1,l,-l],[1,1,l,-l,0,0],[l,-l,0,0,1,1]])
    return thetaetf


def generate_thetayuv():
    thetayuv = np.array([[0.299,0.587,0.114],[-0.147,-0.289,0.436],[0.615,-0.515,-0.100]])
    return thetayuv
    

filename_in = 'Afghan.jpg'
channels_in = 3

block_size = 16

D = generate_dct(block_size*block_size)
psi = sp.linalg.block_diag(D,D,D)

theta = np.concatenate((generate_thetayuv(),generate_thetaetf()),axis=1)
bigtheta = np.kron(theta,np.eye(block_size*block_size))

image = sp.misc.imread(filename_in)
cfa = generate_cfa(image.shape)

image_rows = image.shape[0]
image_cols = image.shape[1]

image_mono = project_image_onto_cfa(image,cfa)
plt.imshow(image_mono,cmap='gray')

image_recon = np.zeros(image.shape)

for i in range(0,int(image_cols/block_size)):
    for j in range(0,int(image_rows/block_size)):
        print('Cycle ' + str(j+i*int(image_rows/block_size)) + ' of ' + str(int(image_cols/block_size)*int(image_rows/block_size)-1))
        
        row_beg = j*block_size
        row_end = (j+1)*block_size
        col_beg = i*block_size
        col_end = (i+1)*block_size
        
        image_mono_block = image_mono[row_beg:row_end,col_beg:col_end]
        cfa_block = cfa[row_beg:row_end,col_beg:col_end]
        
        image_mono_blockv = np.reshape(image_mono_block,(-1,1),'F')
        
        cp_block = list()
        cp_blockv = list()
        cp_blockd = list()
        for k in range(0,channels_in):
            cp_block.append(cfa_block[:,:,k])
            cp_blockv.append(np.reshape(cp_block[k],(-1,1),'F'))
            cp_blockd.append(np.diagflat(cp_blockv[k]))
        
        phi_tuple = tuple(phi_element for phi_element in cp_blockd)
        phi = np.concatenate(phi_tuple,axis=1)
        
        eta = np.dot(phi,psi)
        P = np.dot(eta,bigtheta)

        lasso = Lasso(alpha=0.01)
        lasso.fit(P,image_mono_blockv)
        
        recon = np.dot(np.dot(psi,bigtheta),lasso.coef_);
        
        reconv = list()
        reconvrs = list()
        for k in range(0,channels_in):
            reconv.append(recon[k*block_size**2:(k+1)*block_size**2])
            reconv[k] = np.reshape(reconv[k],(block_size,block_size),'F')
            image_recon[row_beg:row_end,col_beg:col_end,k] = reconv[k]
        
        
plt.imshow(image)
plt.show()
plt.imshow(image_recon)
plt.show()

image_recon = image_recon - np.min(image_recon)
image_recon = image_recon * (1/np.max(image_recon))
image_recon = image*255
image_recon = 255 - image_recon

print(np.min(image_recon))
print(np.max(image_recon))

plt.imshow(image_recon)
plt.show()
