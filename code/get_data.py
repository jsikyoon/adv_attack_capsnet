import os,sys,gzip
import numpy as np
import tensorflow as tf
from skimage.io import imsave

height=28;
width=28;

def image_save(images,fname,grid_size=(8,8),grid_pad=5):
  grid_h=height*grid_size[0]+grid_pad*(grid_size[0]-1);
  grid_w=width*grid_size[1]+grid_pad*(grid_size[1]-1);
  img_grid=np.zeros((grid_h,grid_w,3),dtype=np.uint8);
  for i,img in enumerate(images):
    if i >= grid_size[0]*grid_size[1]:
      break;
    img[img<0.0]=0.0;
    img[img>1.0]=1.0;
    img=np.reshape(img,[height,width,1]);
    img=np.concatenate([img,img,img],2);
    #img = (((img + 1.0) * 0.5) * 255.0).astype(np.uint8)
    img = (img * 255.0).astype(np.uint8)
    row = (i // grid_size[0]) * (height + grid_pad)
    col = (i % grid_size[1]) * (width + grid_pad)
    img_grid[row:row + height, col:col + width] = img
  imsave(fname,img_grid);

