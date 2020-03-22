# coding: utf-8, created 2019-05-15 14:10 EDT by Rajat Verma
"""
Digitize safety belt survey forms using custom scanned forms (hardcoded)
 - Image specs: letter size with dpi=300 i.e. size=(2550,3300)px
 - File path & naming: "./img/scan%d.jpg" % image_num
"""
# packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

# constants
MAP = {'not_observable': [0], 'commercial': [1], 'veh_type': range(3,7),
       'phone_use': range(8,12), 'driver_belt': range(13,15),
       'driver_age': range(16,19), 'driver_gender': range(20,22),
       'driver_race': range(23,26), 'pass_belt': range(27,29),
       'pass_age': range(30,34), 'pass_gender': range(35,37),
       'pass_race': range(38,41)} # row numbers of questions & their options
NROW, NCOL = 41, 10 # no. of grid rows & cols

# grid/layout dimensions
MARGINS = (0, 80) # crop image by these symmetric margins (top, left) (px)
MIN_BOX_AREA = 6e6 # min area for filtering form's bounding box (px^2)
BOX_SIZE = (3060, 2136) # size of the coerced/warped outer box (px)
X0, Y0 = 625, 85 # top-left corner of the grid
DX, DY = 150, 75 # grid step size in x, y directions (px)

# adjustment parameters
ZOOM = 0.25 # default scaling factor for displaying images
BRIGHT = 40 # brightness additive term
CONTRAST = 1.5 # contrast scaling factor
BLUR_SIZE = 5 # size of kernel for blurring (px)
CANNY_THRESH = (40, 100) # thresholds for Canny edge detection
EPSILON = 20 # threshold for polygon approximation from contour (px)
DARK_THRESH = 1e-2 # threshold for classifying marks based on darkness

def blank(img, is_white=False):
    '''create blank canvas of size of a given image'''
    if is_white:
        return 255*np.ones(img.shape, img.dtype)
    else:
        return np.zeros(img.shape, img.dtype)

def show(img, title='Untitled', waitKey=0, scale=ZOOM):
    '''render an image'''
    cv2.imshow(title, cv2.resize(img, (0,0), fx=scale, fy=scale))
    cv2.waitKey(waitKey)
    cv2.destroyAllWindows()

def c(img1, img2):
    '''combine/stack two images horizontally'''
    return np.hstack((img1, img2))

def crop(img, marg=MARGINS):
    '''crop off the offset/margin of an image'''
    return img[marg[0]:img.shape[0]-marg[0], marg[1]:img.shape[1]-marg[1]]

def adjust(img, bright=BRIGHT, contrast=CONTRAST):
    '''adjust brightness and contrast'''
    return cv2.addWeighted(img.copy(), contrast, blank(img), bright, 0)
    
def drawCont(img, draw=True, blur=BLUR_SIZE, bright=BRIGHT, contrast=CONTRAST,
             canny=CANNY_THRESH, iDX=-1, color=255, thick=-1, scale=1):
    '''draw contours on an image'''
    img = cv2.GaussianBlur(img.copy(), (blur, blur), 0)
    img = adjust(img, bright, contrast)
    img = cv2.Canny(img, *canny)
    cont = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cont = sorted(cont, key=cv2.contourArea, reverse=True)
    cont_img = cv2.drawContours(blank(img), cont[:iDX], -1, color, thick)
    if draw:
        show(cont_img, scale=scale)
    else:
        return cont_img, cont
    
def drawGrid(img, X=range(X0, X0+DX*(NCOL+1), DX),
             Y=range(Y0, Y0+DY*(NROW+1), DY), color=0, thick=2):
    '''draw the grid on the straightened image'''
    img = img.copy()
    grid = [((x, 0), (x, img.shape[0])) for x in X] + [
            ((0, y), (img.shape[1], y)) for y in Y]
    for line in grid:
        img = cv2.line(img, *line, color, thick)
    return img

def four_point_transform(img, pts, fixSize=None):
    '''perspective transform the input quadrilateral (4 points) to rectangle'''
    if fixSize is None: # if size of final image is not explicitly given
        pts = np.float32(np.reshape(pts, (4, 2))) # coerce into correct shape
        
        # identify corners of final rectangle using their coordinate geometry
        sum_ = pts.sum(axis=1) # calc the sum of coordinates of the points
        tl = pts[np.argmin(sum_)] # top-left point has the smallest sum
        br = pts[np.argmax(sum_)] # bottom-right point has the largest sum
        diff = np.diff(pts, axis=1) # calc the difference between the points
        tr = pts[np.argmin(diff)] # top-right point has smallest diff.
        bl = pts[np.argmax(diff)] # bottom-left has the largest diff.
    
        # calc new dimensions of new image as the larger of opposite edges
        dist = lambda x: np.linalg.norm(x, 2) # Euclidean distance function
        width = int(max(dist(tl - tr), dist(bl - br))) # top & bottom edges
        height = int(max(dist(tl - bl), dist(tr - br))) # left & right edges
    else: # if size of final image is explicitly given
        height, width = fixSize
    # create the source point matrix for perspective transform
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    # create destination point matrix (to obtain a top-down view of the img)
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]],
                   dtype = np.float32)
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped
    
def straighten(img, fileNum=0, marg=MARGINS, min_box_area=MIN_BOX_AREA,
            box_size=BOX_SIZE):
    '''straighten image to a fixed-size rect using perspective transform'''
    # remove the margins so as to reduce/remove scanner dark areas
    img = crop(img, marg)
    # get the contours of the image
    cont_img, cont = drawCont(img, draw=False)
    # get area of largest contour
    box_area = cv2.contourArea(cont[0])
    # approximate the quadrilateral of the bounding box
    box_poly = cv2.approxPolyDP(cont[0], EPSILON, True)
    # filter for only very large quadrilaterals
    if box_area < min_box_area or len(box_poly) != 4: # oops, bad luck!
        print('Could not fit rectangle of outer box. (img=%s, n=%d) \
              Box area = %.2e' % (fileNum, len(box_poly), box_area))
        warped = None
    else:
        # perspective transform to coerce bounding quad to rectangle
        warped = four_point_transform(img.copy(), box_poly)
    return warped, cont, box_area

def cell(img, i, j):
    '''locate cell using its indices'''
    return(img[Y0+i*DY : Y0+(i+1)*DY, X0+j*DX : X0+(j+1)*DX])

def classify(darks, thresh=DARK_THRESH, plot=False):
    '''assign single value to vars based on darkness matrix & threshold'''
    # throw error if input not in good shape
    if darks.shape != (NROW, NCOL):
        raise ValueError('Bad image shape: (%d, %d)' % darks.shape)
    # initialize a (variable x column) matrix with default value = 0
    maxes = np.zeros((len(MAP.keys()), NCOL), dtype=np.int8)
    # iterate over every column and variable
    for j in range(NCOL):
        for k, (var, iDX) in enumerate(MAP.items()):
            # get local index of max darkness cell for each variable
            iMax = np.argmax(darks[iDX, j])
            maxValue = darks[iDX[iMax], j]
            # assign response (1,2,...) to variable
            if maxValue > thresh:
                maxes[k, j] = iMax + 1
    if plot:
        # plot heatmap of binary classification; used for testing threshold
        plt.imshow(maxes > thresh, cmap='binary', extent=(0, 50, 0, 48))
    return maxes

def digitize(files):
    '''read and convert the data of the scanned files and store in a dataframe'''
    # read the images in grayscale
    imgs = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])
    # initialize the result dataframe
    result = pd.DataFrame()
    # for all images
    for imgNum, (img, file) in enumerate(zip(imgs, files)):
        # perspective transform the bounding box of the image
        img, cont, box_area = straighten(img, file)
        if img is None:
            continue
        # cleanse the image
        img = adjust(img)
#        gridImg = drawGrid(adjust(img)) # create the grid
        # iterate over the grid for mark detection
        darks = np.zeros((NROW, NCOL), dtype=np.float32)
        for i in range(NROW):
            for j in range(NCOL):
                # get total value of darkness (negative of brightness)
                darks[i, j] = np.sum(255 - cell(img, i, j)) / (255*DX*DY)
        # plot the (grayscale) heatmap of total darkness of the cells
#        plt.imshow(darks, cmap='binary', extent=(0, 50, 0, 41)
        # classify the form using the current darkness matrix
        classified = classify(darks).transpose()
        # convert the data array to the more readable dataframe
        df = pd.DataFrame(classified, columns=[x.upper() for x in MAP.keys()])
        # add vehicle/observation number as row index
        df.insert(0, 'OBS_NUM', imgNum * NCOL + np.arange(1, NCOL+1))
        # remove the all-zero rows (i.e. empty forms)
        df = df[(df.T[1:] != 0).any()]
        # append the result of this iteration to the main result dataframe
        result = pd.concat([result, df], axis=0)
    return result

if __name__ == '__main__':
    path = 'test/Sample Site 1'
    
    # digitize one file
    result = digitize('%s/scan0.jpg' % path)

    # digitize specific files
    result = digitize(['%s/%d.jpg' % (path, x) for x in [0,3,4]])
    
    # digitize all files in the folder
    path = 'test/Sample Site 1'
    result = digitize(list(glob('%s/*[0-9].jpg' % path)))

    # save the site data to a CSV file
    result.to_csv('%s/Output.csv' % path, index=False)