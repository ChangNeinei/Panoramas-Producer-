import numpy as np
import scipy.io as sio
import skimage.feature
from skimage.feature import (corner_peaks, corner_harris, corner_fast)
import matplotlib.pyplot as plt
from scipy import misc
from skimage import color
import random
import math

# Q2.1
# create a 2 x nbits sampling of integers from to to patchWidth^2
# read BRIEF paper for different sampling methods
def makeTestPattern(patchWidth, nbits):
    compareX = [random.randint(0, patchWidth ** 2 - 1) for i in range(nbits)]
    compareY = [random.randint(0, patchWidth ** 2 - 1) for i in range(nbits)]
    #np.save('testPattern_compareX.npy', np.array(compareX))
    #np.save('testPattern_compareY.npy', np.array(compareY))
    sio.savemat('testPattern.mat',{'compareX':compareX,'compareY':compareY})
    return compareX, compareY

# Q2.2
# im is 1 channel image, locs are locations
# compareX and compareY are idx in patchWidth^2
# should return the new set of locs and their descs
def computeBrief(im,locs,compareX,compareY,patchWidth=9):
    h, w = im.shape #139, 98
    # YOUR CODE HERE
    valid_patch = math.floor(patchWidth/2)
    desc = []
    locs_new = []
    for pts in locs:
        if (check_in_img(h, w, pts)):
            x, y = pts
            patch = im[(x - valid_patch) : (x + valid_patch + 1), (y - valid_patch) : (y + valid_patch + 1)]
            descript = T_descript(patch, compareX, compareY)
            locs_new.append(pts)
            desc.append(descript)
    print(desc)
    return np.array(locs_new), np.array(desc)

def check_in_img(h, w, points, patchWidth = 9):
    patch = math.floor(patchWidth / 2)
    x, y = points
    if (x - patch) >= 0 and (y - patch) >= 0 and (x + patch) < h and (y + patch) < w:
        return True
    else:
        return False

def T_descript(matrix, compX, compY):
    m = matrix.flatten()
    diff = m[compX] - m[compY]
    return (diff < 0) * np.ones(len(compX))

# Q2.3
# im is a 1 channel image
# locs are locations
# descs are descriptors
# if using Harris corners, use a sigma of 1.5
def briefLite(im):
    # YOUR CODE HERE
    #X = np.load('testPattern_compareX.npy')
    #Y = np.load('testPattern_compareY.npy')
    pattern = sio.loadmat('testPattern.mat')
    X = pattern.get('compareX')[0,:]
    Y = pattern.get('compareY')[0,:]
    #keypoints1 = corner_peaks(corner_harris(im, method = "eps"), min_distance=1)
    keypoints1 = corner_peaks(corner_fast(im), min_distance=1)
    locs, desc = computeBrief(im,keypoints1,X,Y,patchWidth=9)
    return locs, desc

# Q 2.4
def briefMatch(desc1,desc2,ratio=0.8):
    # okay so we say we SAY we use the ratio test
    # which SIFT does
    # but come on, I (your humble TA), don't want to.
    # ensuring bijection is almost as good
    # maybe better
    # trust me
    matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True)
    return matches

def plotMatches(im1,im2,matches,locs1,locs2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r')
    plt.show()
    return

def testMatch():
    # YOUR CODE HERE
    I_1 = misc.imread('../data/model_chickenbroth.jpg')
    I_2 = misc.imread('../data/chickenbroth_01.jpg')
    img1 = color.rgb2gray(I_1)
    img2 = color.rgb2gray(I_2)
    locs1, desc1 = briefLite(img1)
    locs2, desc2 = briefLite(img2)
    matches = briefMatch(desc1,desc2,ratio=0.8)
    print(len(matches))
    plotMatches(img1,img2,matches,locs1,locs2)
    return


# Q 2.5
# we're also going to use this to test our
# extra-credit rotational code
def briefRotTest(briefFunc=briefLite):
    # you'll want this
    import skimage.transform
    # YOUR CODE HERE
    I_1 = misc.imread('../data/model_chickenbroth.jpg')
    img1 = color.rgb2gray(I_1)
    locs1, desc1 = briefLite(img1)
    row, col = img1.shape
    center_row = row / 2
    center_col = col / 2
    correct = []
    for i in range(0, 0, 10):
        #correct = 0
        img2 = skimage.transform.rotate(img1, i)
        locs2, desc2 = briefLite(img2)
        matches = briefMatch(desc1,desc2,ratio=0.8)
        '''
        cor_locs = rotate(center_row, center_col, i, locs1[matches[:,0]])
        dis = np.subtract(cor_locs, locs2[matches[:,1]])
        dis = np.sum(dis**2, 1)
        print(dis)
        a = dis < 30
        correct_matches = []
        for j in range(len(a)):
            if (a[j]):
                correct_matches.append(matches[j])
        '''
        correct.append(len(matches))
        plotMatches(img1,img2,np.array(matches),locs1,locs2)
    plt.bar(range(0,37), correct)
    plt.show()
    return

def rotate(originRow, originCol, angle, Points):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    origin = np.repeat(np.expand_dims([originRow, originCol], 0), len(Points), 0)
    scale = np.subtract(Points, origin)
    rotX = [math.cos(angle * math.pi / 180), - math.sin(angle * math.pi / 180)]
    rotY = [math.sin(angle * math.pi / 180), math.cos(angle * math.pi / 180)]
    new_Row = np.sum(scale * np.expand_dims(rotX, 0), 1)
    new_Col = np.sum(scale * np.expand_dims(rotY, 0), 1)
    new_Points = np.concatenate([np.expand_dims(new_Row, 1), np.expand_dims(new_Col, 1)], 1)
    return np.add(new_Points, origin)
 
# Q2.6
# YOUR CODE HERE


# put your rotationally invariant briefLite() function here
def briefRotLite(im):
    locs, desc = None, None
    # YOUR CODE HERE
    return locs, desc


'''
def main():
    makeTestPattern(9, 256)
    #testMatch()
    #briefRotTest(briefFunc=briefLite)


if __name__ == "__main__":
    main()
'''
