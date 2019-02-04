from q4 import *
from q3 import *
from q2 import *
import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import skimage.io

# Q 4.1
# load images into img1
# and img2
# compute H2to1
# please use any feature method you like that gives
# good results
img1 = skimage.io.imread('../data/incline_L.png')
img2 = skimage.io.imread('../data/incline_R.png')
'''
img1 = skimage.io.imread('../data/pnc0.png')
img2 = skimage.io.imread('../data/pnc1.png')
'''
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
bestH2to1 = bestH

panoImage = imageStitching(img1,img2,bestH2to1)

plt.subplot(1,2,1)
plt.imshow(img1)
plt.title('incline_L')
plt.subplot(1,2,2)
plt.title('incline_R')
plt.imshow(img2)
plt.figure()
plt.imshow(panoImage)
plt.show()

# Q 4.2
panoImage2= imageStitching_noClip(img1,img2,bestH2to1)

plt.subplot(2,1,1)
plt.imshow(panoImage)
plt.subplot(2,1,2)
plt.imshow(panoImage2)
plt.show()


# Q 4.3
panoImage3 = generatePanorama(img1, img2)

# Q 4.4 (EC)
# Stitch your own photos with your code
'''
cmuL = skimage.io.imread('../data/CMUL.jpg')
l = resize(cmuL, (612, 816, 3))
cmuR = skimage.io.imread('../data/CMUR.jpg')
r = resize(cmuR, (612, 816, 3))
panoImageCMU = generatePanorama(l, r)
M = np.array([[1, 0, 0], [0, 1, 2], [0, 0, 1]])
warp_im1 = skimage.transform.warp(l, inv(M), output_shape = (820, 2000))
plt.subplot(1,2,1)
plt.imshow(l)
plt.title('cmuLeft')
plt.subplot(1,2,2)
plt.title('cmuRight')
plt.imshow(r)
plt.figure()
plt.imshow(warp_im1 + panoImageCMU)
plt.show()
'''

# Q 4.5 (EC)
# Write code to stitch multiple photos
# see http://www.cs.jhu.edu/~misha/Code/DMG/PNC3/PNC3.zip
# for the full PNC dataset if you want to use that
if False:
    imgs = [skimage.io.imread('../PNC3/src_000{}.png'.format(i)) for i in range(7)]
    panoImage4 = generateMultiPanorama(imgs)
    plt.imshow(panoImage4)
    plt.show()
