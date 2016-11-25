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

import math
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


def project_image_through_cfa(image, cfa):
    projection = np.multiply(image, cfa)
    return np.sum(projection,axis=2)


def generate_dct(N):
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            D[k,n] = np.cos((np.pi/N)*(n+0.5)*k)
    
    D[0,:] = (1/np.sqrt(2))*D[0,:]
    D[:,:] = np.sqrt(2/N)*D[:,:]
    
    return D


def generate_thetaetf():
    l = (1+np.sqrt(5))/2
    thetaetf = (1/np.sqrt(1+l**2))*np.array([[0,0,1,1,l,-l], [1,1,l,-l,0,0], [l,-l,0,0,1,1]])
    return thetaetf


def generate_rgbtoyuv():
    thetargb = np.array([[0.299,0.587,0.114],[-0.147,-0.289,0.436],[0.615,-0.515,-0.100]])
    return thetargb


def generate_yuvtorgb():
    thetayuv = np.array([[1.000,0.000,1.140],[1.000,-0.395,-0.581],[1.000,2.032,0.000]])
    return thetayuv

def normalize(A):
    return 1/255*A

def denormalize(A):
    return 255*A
    

filename_in = 'Afghan.jpg'
image = sp.misc.imread(filename_in)

image_channels = 3
image_rows = image.shape[0]
image_cols = image.shape[1]
image_size = image_rows*image_cols

cfa = generate_cfa(image.shape,image_channels)
y = project_image_through_cfa(image,cfa)

image_recon = np.zeros((image_rows,image_cols,image_channels))


block_size = 16


lasso = Lasso(alpha=0.01)


psi = generate_dct(block_size**2)
bigpsi = sp.kron(psi,np.eye(image_channels))

thetayuv = generate_yuvtorgb()
thetaetf = generate_thetaetf()
theta = np.hstack((thetayuv,thetaetf))
bigtheta = np.kron(theta,np.eye(block_size**2))


numofblocks_cols = math.floor(image_cols/block_size)
numofblocks_rows = math.floor(image_rows/block_size)


for col in range(0,numofblocks_cols):
    for row in range(0,numofblocks_rows):
        
        cycle_curr = row + col*numofblocks_rows
        cycle_total = numofblocks_cols*numofblocks_rows-1
        
        print('Cycle ' + str(cycle_curr) + ' of ' + str(cycle_total))
        
        if col != numofblocks_cols-1:
            col_beg = col*block_size
            col_end = (col+1)*block_size
        else:
            col_beg = col*block_size
            col_end = image_cols
        
        if row != numofblocks_rows-1:
            row_beg = row*block_size
            row_end = (row+1)*block_size
        else:
            row_beg = row*block_size
            row_end = image_rows
        
        block_cols = col_end - col_beg + 1
        block_rows = row_end - row_beg + 1
        
        y_block = y[row_beg:row_end,col_beg:col_end]
        y_blockvec = np.reshape(y_block,(-1,1),'F')
        
        cfa_block = cfa[row_beg:row_end,col_beg:col_end]
        cfa_blockdiag = list()
        
        for channel in range(0,image_channels):
            temp = cfa_block[:,:,channel]            
            temp = np.reshape(temp,(-1,1),'F')                                   
            cfa_blockdiag.append(np.diagflat(temp))
        
        phi_tuple = tuple(element for element in cfa_blockdiag)
        phi = np.hstack(phi_tuple)
        
        eta = np.dot(phi,bigpsi)
        P = np.dot(eta,bigtheta)
        Pprime = np.dot(bigpsi,bigtheta)

        lasso.fit(P,y_blockvec)
        x = lasso.coef_
        
        image_recon_block = np.dot(Pprime,x)
        
        for channel in range(0,image_channels):
            temp = image_recon_block[channel*block_size**2:(channel+1)*block_size**2]
            temp = np.reshape(temp,(block_size,block_size),'F')
            image_recon[row_beg:row_end,col_beg:col_end,channel] = temp
        

plt.imshow(image)
plt.show()
plt.imshow(y,cmap='gray')
plt.show()
plt.imshow(image_recon)
plt.show()

plt.hist(np.reshape(y,(-1,1)))
plt.show()

print(np.min(image_recon[:,:,1]))
print(np.max(image_recon[:,:,1]))


image_recon = image_recon - np.min(image_recon)
image_recon = image_recon * (1/np.max(image_recon))
image_recon = image*255
image_recon = 255 - image_recon

print(np.min(image_recon))
print(np.max(image_recon))

plt.imshow(image_recon)
plt.show()
