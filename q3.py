import numpy as np
import skimage.color
import skimage.io
from scipy import linalg
import math
import random
import q2
from scipy import misc
from skimage import color
import matplotlib.pyplot as plt
import scipy.io
from numpy.linalg import inv
from PIL import Image
from resizeimage import resizeimage
from skimage.transform import resize

# Q 3.1
def computeH(l1, l2):
    #H2to1 = np.eye(3)
    len_l2 = len(l2)
    # YOUR CODE HERE
    pad = np.ones((len_l2, 1), dtype=int)
    neg_l2 = l2 * (- 1)
    first_three = np.insert(neg_l2, 2, -1, 1)
    zero = np.zeros((len_l2, 3), dtype=int)
    x1 = l1[:,0]
    y1 = l1[:,1]
    x1_x1 = np.repeat(np.expand_dims(x1, 0), 3, 0).T
    y1_y1 = np.repeat(np.expand_dims(y1, 0), 3, 0).T
    last_three_x = (first_three * (-1)) * x1_x1
    last_three_y = (first_three * (-1)) * y1_y1
    upper = np.concatenate((first_three, zero, last_three_x), 1)
    lower = np.concatenate((zero, first_three, last_three_y), 1)
    A = np.concatenate((upper, lower), 1).reshape(2 * len_l2, 9)
    U,sigma,VT=linalg.svd(A)
    h = VT[-1]
    H2to1 = h.reshape(3,3)
    return H2to1

# Q 3.2
def computeHnorm(l1, l2):
    #H2to1 = np.eye(3)
    mean_points_l1 = np.mean(l1, axis=0)
    mean_points_l2 = np.mean(l2, axis=0)
    origin_l1 = np.repeat(np.expand_dims(mean_points_l1, 0), len(l1), 0)
    origin_l2 = np.repeat(np.expand_dims(mean_points_l2, 0), len(l2), 0)
    new_l1 = l1 - origin_l1
    new_l2 = l2 - origin_l2
    scale_l1 = max(np.sum((new_l1 ** 2), 1)) / 2
    scale_l2 = max(np.sum((new_l2 ** 2), 1)) / 2
    new_l1 = new_l1 / math.sqrt(scale_l1)
    new_l2 = new_l2 / math.sqrt(scale_l2)
    # YOUR CODE HERE
    H2to1 = computeH(new_l1, new_l2)
    #print(H2to1)
    return H2to1

# Q 3.3
def computeHransac(locs1, locs2):
    # YOUR CODE HERE
    s = 4 # sample points
    p = 0.99 # accuracy
    e = 0.8 # error ration
    N = np.log(1-p) / np.log(1-np.power((1-e), s))
    threshold = 2
    maxInlier = 0
    h1 = np.concatenate([locs1, np.ones((len(locs1), 1))], 1)
    h2 = np.concatenate([locs2, np.ones((len(locs2), 1))], 1)
    for i in range(1000): #math.ceil(N)
        # Sample 4 points
        rand_index = [random.randint(0, len(locs1) - 1) for n in range(s)]
        sample_locs1 = locs1[rand_index]
        sample_locs2 = locs2[rand_index]
        tempH = computeH(sample_locs1, sample_locs2)
        locs2_afterH = np.matmul(tempH, h2.T)
        locs2_Hnom = (locs2_afterH / np.repeat([locs2_afterH[-1, :]], 3, axis = 0))
        dist = np.sum(np.subtract(locs2_Hnom.T, h1) ** 2, 1)
        inliers = (dist < threshold ** 2) * np.ones(len(dist))
        inlier_sum = np.sum(inliers)
        if (inlier_sum > maxInlier):
            maxInlier = inlier_sum
            bestinlier = inliers
    best_index = np.argwhere(bestinlier == 1).flatten()
    best_locs1 = locs1[best_index]
    best_locs2 = locs2[best_index]
    bestH2to1 = computeH(best_locs1, best_locs2)
    return bestH2to1, bestinlier


# Q3.4
# skimage.transform.warp will help
def compositeH(H2to1, template, img):
    compositeimg = []
    # YOUR CODE HERE
    img_gray = color.rgb2gray(img)
    hp_desk = skimage.transform.warp(img, inv(H2to1), output_shape = template.shape)
    mask = (hp_desk == 0)
    desk_background = mask * template
    plt.imshow(desk_background + hp_desk)
    plt.show()
    return compositeimg


def HarryPotterize():
    # we use ORB descriptors but you can use something else
    from skimage.feature import ORB,match_descriptors
    # YOUR CODE HERE
    I_1 = misc.imread('../data/cv_desk.png')
    I_2 = misc.imread('../data/cv_cover.jpg') / 255.
    I_3 = misc.imread('../data/hp_cover.jpg')
    img1 = skimage.color.rgba2rgb(I_1)
    desk = color.rgb2gray(img1)
    cover = color.rgb2gray(I_2)
    hp = color.rgb2gray(I_3)
    
    desk = np.transpose(desk)
    cover = np.transpose(cover)
    hp = np.transpose(hp)

    locs1, desc1 = q2.briefLite(desk)
    locs2, desc2 = q2.briefLite(cover)
    matches = q2.briefMatch(np.array(desc1), np.array(desc2))
    l1 = np.asarray(locs1)
    l2 = np.asarray(locs2)
    bestH, inliers = computeHransac(l1[matches[:, 0]], l2[matches[:, 1]])

    #compositeimg = skimage.transform.warp(np.transpose(cover), inv(bestH), output_shape = np.transpose(desk).shape)
    
    new_hp = resize(I_3, I_2.shape)
    compositeH(bestH, img1, new_hp)
    #plt.imshow()
    #plt.show()
    return

'''
def main():
    l1 = np.array([[10,10],[21,21],[20,20],[20,15]]) 
    l2 = np.array([[0,0],[11,11],[10,10],[10,5]])
    H = computeH(l1, l2)
    #print(H)
    HarryPotterize()

if __name__ == "__main__":
    main()
'''