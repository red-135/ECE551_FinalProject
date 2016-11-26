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

from spgl1 import spgl1

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


def generate_dct_ortho(N):
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            D[k,n] = np.cos((np.pi/N)*(n+0.5)*k)
    
    D[0,:] = (1/np.sqrt(2))*D[0,:]
    D[:,:] = np.sqrt(2/N)*D[:,:]
    
    return D


def generate_idct_ortho(N):
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            D[k,n] = np.cos((np.pi/N)*n*(k+0.5))
    
    D[:,0] = (1/np.sqrt(2))*D[:,0]
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


def pad_to_blocksize(image, channels, blocksize):
    rows = image.shape[0]
    cols = image.shape[1]
    
    rows_extended = math.ceil(rows/blocksize)*blocksize
    cols_extended = math.ceil(cols/blocksize)*blocksize
    
    image_new = np.zeros((rows_extended,cols_extended,channels))

    if image.ndim == 3:    
        image_new[0:rows,0:cols,:] = image
    else:
        image_new[0:rows,0:cols,:] = image[:,:,np.newaxis]
    
    return image_new


def pad_to_blocksize_with_offset(image, channels, blocksize, offset):    
    rows = image.shape[0]
    cols = image.shape[1]
    
    rows_extended = math.ceil((rows+offset)/blocksize)*blocksize
    cols_extended = math.ceil((cols+offset)/blocksize)*blocksize
    
    image_new = np.zeros((rows_extended,cols_extended,channels))
    
    if image.ndim == 3:    
        image_new[offset:rows+offset,offset:cols+offset,:] = image
    else:
        image_new[offset:rows+offset,offset:cols+offset,:] = image[:,:,np.newaxis]
    
    return image_new


filename_in = 'Afghan.jpg'
image_channels = 3
block_size = 16


image_orig = sp.misc.imread(filename_in)

image_orig_rows = image_orig.shape[0]
image_orig_cols = image_orig.shape[1]

cfa_orig = generate_cfa(image_orig.shape,image_channels)
y_orig = project_image_through_cfa(image_orig,cfa_orig)


image = pad_to_blocksize(image_orig, image_channels, block_size)
cfa = pad_to_blocksize(cfa_orig, image_channels, block_size)
y = pad_to_blocksize(y_orig, 1, block_size)

image_rows = image.shape[0]
image_cols = image.shape[1]
image_size = image_rows*image_cols

image_recon_final = np.zeros((image_orig_rows,image_orig_cols,image_channels))
image_recon_final[:] = np.NAN
y_recon_final = np.zeros((image_orig_rows,image_orig_cols))
y_recon_final[:] = np.NAN


# UNCOMMENT TO USE LASSO
# lasso = Lasso(alpha=0.01)


# OLD METHOD: 1D DCT
#psi = generate_idct_ortho(block_size**2)

# NEW METHOD: 2D DCT
A = generate_idct_ortho(block_size)
B = generate_dct_ortho(block_size)
psi = sp.kron(B.T,A)

bigpsi = sp.kron(np.eye(image_channels),psi)

thetayuv = generate_yuvtorgb()
thetaetf = generate_thetaetf()
theta = np.hstack((thetayuv,thetaetf))
bigtheta = np.kron(theta,np.eye(block_size**2))


for offset in [0,4,8,12]:
    image = pad_to_blocksize_with_offset(image_orig, image_channels, block_size, offset)
    cfa = pad_to_blocksize_with_offset(cfa_orig, image_channels, block_size, offset)
    y = pad_to_blocksize_with_offset(y_orig, 1, block_size, offset)
    
    image_rows = image.shape[0]
    image_cols = image.shape[1]
    image_size = image_rows*image_cols
    
    image_recon = np.zeros((image_rows,image_cols,image_channels))
    y_recon = np.zeros((image_rows,image_cols))
    
    numofblocks_cols = math.floor(image_cols/block_size)
    numofblocks_rows = math.floor(image_rows/block_size)    
    
    for col in range(0,numofblocks_cols):
        for row in range(0,numofblocks_rows):
            
            cycle_curr = row + col*numofblocks_rows
            cycle_total = numofblocks_cols*numofblocks_rows-1
            
            print('Cycle ' + str(cycle_curr) + ' of ' + str(cycle_total))
            
            col_beg = col*block_size
            col_end = (col+1)*block_size
            row_beg = row*block_size
            row_end = (row+1)*block_size
            
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
    
            # UNCOMMENT TO USE LASSO
            # lasso.fit(P,y_blockvec.flatten())
            # x = lasso.coef_
           
            x,resid,grad,info = spgl1.spg_bp(P,y_blockvec.flatten())
                    
            print('L0 Norm: ' + str(np.linalg.norm(x,ord=0)) + ' of ' + str(x.size))
            print('Sparsity: ' + str(np.linalg.norm(x,ord=0)/x.size*100) + ' %')
            
            image_recon_block = np.dot(Pprime,x)
            y_recon_block = np.dot(P,x)
            
            for channel in range(0,image_channels):
                temp1 = image_recon_block[channel*block_size**2:(channel+1)*block_size**2]
                temp1 = np.reshape(temp1,(block_size,block_size),'F')
                image_recon[row_beg:row_end,col_beg:col_end,channel] = temp1
                
                temp2 = np.reshape(y_recon_block,(block_size,block_size),'F')
                y_recon[row_beg:row_end,col_beg:col_end] = temp2
    
    image_temp = image_recon[offset:offset+image_orig_rows,offset:offset+image_orig_cols,:]
    y_temp = y_recon[offset:offset+image_orig_rows,offset:offset+image_orig_cols]
    
    for channel in range(0,image_channels):
        image_recon_final[:,:,channel] = np.nanmedian(np.dstack((image_recon_final[:,:,channel],image_temp[:,:,channel])),axis=2)
        y_recon_final[:,:] = np.nanmedian(np.dstack((y_recon_final[:,:],y_temp)),axis=2)
    
    plt.imshow(image_temp.astype(np.uint8))
    plt.show()
    plt.imshow(y_temp,cmap='gray')
    plt.show()


image_final = image_recon_final[0:image_orig_rows,0:image_orig_cols]
image_final[image_final<0] = 0
image_final[image_final>255] = 255

y_recon = y_recon_final

plt.imshow(image_orig.astype(np.uint8))
plt.show()
plt.imshow(image_final.astype(np.uint8))
plt.show()
plt.imshow(y_orig,cmap='gray')
plt.show()
plt.imshow(y_recon,cmap='gray')
plt.show()

plt.hist(image_orig.flatten(),255)
plt.show()
plt.hist(image_final.flatten(),255)
plt.show()

print('Error in y: ' + str(np.linalg.norm(y_orig - y_recon)))
print('Error in Image: ' + str(np.linalg.norm(image_orig - image_final)))

print('Minimum: ' + str(np.min(image_final[:,:,1])))
print('Maximum: ' + str(np.max(image_final[:,:,1])))

plt.imshow(np.sum(np.fabs(image_final-image_orig),axis=2).astype(np.uint8),cmap='gray')
plt.show()

image_reconmed = np.zeros(image_final.shape)
for channel in range(0,image_channels):
    image_reconmed[:,:,channel] = sp.signal.medfilt(image_final[:,:,channel])

plt.imshow(image_reconmed.astype(np.uint8))
plt.show()


import winsound
winsound.Beep(300,2000)