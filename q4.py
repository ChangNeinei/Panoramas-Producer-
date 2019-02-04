import numpy as np
from q2 import *
from q3 import *
import skimage.color

# you may find this useful in removing borders
# from pnc series images (which are RGBA)
# and have boundary regions
def clip_alpha(img):
    img[:,:,3][np.where(img[:,:,3] < 1.0)] = 0.0
    return img 

# Q 4.1
# this should be a less hacky version of
# composite image from Q3
# img1 and img2 are RGB-A
# warp order 0 might help
# try warping both images then combining
# feel free to hardcode output size
def imageStitching(img1, img2, H2to1):
    # YOUR CODE HERE
    img1_gray = color.rgb2gray(img1)
    img1_T = np.transpose(img1_gray)
    panoImg = skimage.transform.warp(img2, inv(H2to1), output_shape = np.transpose(img1_T).shape)
    return panoImg


# Q 4.2
# you should make the whole image fit in that width
# python may be inv(T) compared to MATLAB
def imageStitching_noClip(img1, img2, H2to1, panoWidth=1280):
    farest = np.array([[panoWidth], [0], [1]])
    x, y, z = np.matmul(H2to1, farest)
    x = x/z
    y = y/z
    M = np.array([[1, 0, 0], [0, 1, -y], [0, 0, 1]])
    warp_im1 = skimage.transform.warp(img1, inv(M), output_shape = (947, 1700))
    warp_im2 = skimage.transform.warp(img2, np.dot(inv(H2to1), inv(M)), output_shape = (947, 1700))
    #generatePanorama(warp_im1, warp_im2)
    mask_1 = (warp_im1 != 0) * np.ones(warp_im1.shape)
    mask_2 = (warp_im2 != 0) * np.ones(warp_im2.shape)
    mask = mask_2 * mask_1
    mask_new = np.ones(mask.shape) - mask
    panoImg = warp_im1 + mask_new * warp_im2
    return panoImg

# Q 4.3
# should return a stitched image
# if x & y get flipped, np.flip(_,1) can help
def generatePanorama(img1, img2):
    panoImage = None
    # YOUR CODE HERE
    left = color.rgb2gray(img1)
    right = color.rgb2gray(img2)
    l = np.transpose(left)
    r = np.transpose(right)
    locs1, desc1 = briefLite(l)
    locs2, desc2 = briefLite(r)
    matches = briefMatch(np.array(desc1), np.array(desc2))
    l1 = np.asarray(locs1)
    l2 = np.asarray(locs2)
    bestH, inliers = computeHransac(l1[matches[:, 0]], l2[matches[:, 1]])
    #panoImage = imageStitching(img1, img2, bestH)
    panoImage = imageStitching_noClip(img1, img2, bestH, panoWidth=1280)
    return panoImage

# Q 4.5
# I found it easier to just write a new function
# pairwise stitching from right to left worked for me!
def generateMultiPanorama(imgs):
    panoImage = None
    # YOUR CODE HERE
    
    return panoImage
'''
def main():
    I_1 = misc.imread('../data/incline_L.png')
    I_2 = misc.imread('../data/incline_R.png')
    generatePanorama(I_1, I_2)

if __name__ == "__main__":
    main()
'''


