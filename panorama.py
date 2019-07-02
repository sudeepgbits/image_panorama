""" Panoramas

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file.

    2. DO NOT import any other libraries aside from those that we provide.
    You may not import anything else, and you should be able to complete
    the assignment with the given libraries (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the provided virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""

import numpy as np
import scipy as sp
import cv2


def getImageCorners(image):
    """Return the x, y coordinates for the four corners bounding the input
    image and return them in the shape expected by the cv2.perspectiveTransform
    function. (The boundaries should completely encompass the image.)

    Parameters
    ----------
    image : numpy.ndarray
        Input can be a grayscale or color image

    Returns
    -------
    numpy.ndarray(dtype=np.float32)
        Array of shape (4, 1, 2).  The precision of the output is required
        for compatibility with the cv2.warpPerspective function.

    Notes\
    -----
        (1) Review the documentation for cv2.perspectiveTransform (which will
        be used on the output of this function) to see the reason for the
        unintuitive shape of the output array.

        (2) When storing your corners, they must be in (X, Y) order -- keep
        this in mind and make SURE you get it right.
    """
    n = image.shape[1]
    m = image.shape[0]
    #print n
    #print m
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE
    corners[0,0,0] = np.float32(0)
    corners[0,0,1] = np.float32(0)
    corners[1,0,0] = np.float32(n)
    corners[1,0,1] = np.float32(0)
    corners[2,0,0] = np.float32(n)
    corners[2,0,1] = np.float32(m)
    corners[3,0,0] = np.float32(0)
    corners[3,0,1] = np.float32(m)
    
    #print ('image corners =' + str(corners))
    #cv2.imshow('image1',image)
    #cv2.waitKey(0)
    return corners
    


def findMatchesBetweenImages(image_1, image_2, num_matches):
    """Return the top list of matches between two input images.

    Parameters
    ----------
    image_1 : numpy.ndarray
        The first image (can be a grayscale or color image)

    image_2 : numpy.ndarray
        The second image (can be a grayscale or color image)

    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can.

    Returns
    -------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches : list<cv2.DMatch>
        A list of the top num_matches matches between the keypoint descriptor
        lists from image_1 and image_2

    Notes
    -----
        (1) You will not be graded for this function.
    """
    feat_detector = cv2.ORB(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),
                     key=lambda x: x.distance)[:num_matches]
    return image_1_kp, image_2_kp, matches


def findHomography(image_1_kp, image_2_kp, matches):
    """Returns the homography describing the transformation between the
    keypoints of image 1 and image 2.

        ************************************************************
          Before you start this function, read the documentation
                  for cv2.DMatch, and cv2.findHomography
        ************************************************************

    Follow these steps:

        1. Iterate through matches and store the coordinates for each
           matching keypoint in the corresponding array (e.g., the
           location of keypoints from image_1_kp should be stored in
           image_1_points).

            NOTE: Image 1 is your "query" image, and image 2 is your
                  "train" image. Therefore, you index into image_1_kp
                  using `match.queryIdx`, and index into image_2_kp
                  using `match.trainIdx`.

        2. Call cv2.findHomography() and pass in image_1_points and
           image_2_points, using method=cv2.RANSAC and
           ransacReprojThreshold=5.0.

        3. cv2.findHomography() returns two values: the homography and
           a mask. Ignore the mask and return the homography.

    Parameters
    ----------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the first image

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the second image

    matches : list<cv2.DMatch>
        A list of matches between the keypoint descriptor lists

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2
    """
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE.
    i = 0
    for mat in matches:
        
        #print ('loop =' + str(i))
        image_1_points[i,:,:] = np.float32(image_1_kp[mat.queryIdx].pt)
        
        image_2_points[i,:,:] = np.float32(image_2_kp[mat.trainIdx].pt)
        i = i + 1
    
    M, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC,5.0)
    
    #print M
    #print image_1_points
    return M


def getBoundingCorners(corners_1, corners_2, homography):
    """Find the coordinates of the top left corner and bottom right corner of a
    rectangle bounding a canvas large enough to fit both the warped image_1 and
    image_2.

    Given the 8 corner points (the transformed corners of image 1 and the
    corners of image 2), we want to find the bounding rectangle that
    completely contains both images.

    Follow these steps:

        1. Use the homography to transform the perspective of the corners from
           image 1 (but NOT image 2) to get the location of the warped
           image corners.

        2. Get the boundaries in each dimension of the enclosing rectangle by
           finding the minimum x, maximum x, minimum y, and maximum y.

    Parameters
    ----------
    corners_1 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 1

    corners_2 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 2

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_min, y_min) -- the coordinates of the
        top left corner of the bounding rectangle of a canvas large enough to
        fit both images (leave them as floats)

    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_max, y_max) -- the coordinates of the
        bottom right corner of the bounding rectangle of a canvas large enough
        to fit both images (leave them as floats)

    Notes
    -----
        (1) The inputs may be either color or grayscale, but they will never
        be mixed; both images will either be color, or both will be grayscale.

        (2) Python functions can return multiple values by listing them
        separated by commas.

        Ex.
            def foo():
                return [], [], []
    """
    # WRITE YOUR CODE HERE
    i =0
    #print ('corners2 =' + str(corners_2))
    
    #print ('homography = ' +str(homography))
    #print len(corners_1)
    
    corner1_persptransform = cv2.perspectiveTransform(corners_1, homography)
    
    #print ('corner1_perpective = ' +str(corner1_persptransform))
    #print ('corner1_perpective1 = ' +str(corner1_persptransform[:,0,1]))
    min = np.float64([0, 0])
    max = np.float64([0, 0])
    
    
    #print corner1_persptransform
    
    cor1_minx = np.float64(np.min(corner1_persptransform[:,0,0]))
    cor2_minx = np.float64(np.min(corners_2[:,0,0]))
    cor1_miny = np.float64(np.min(corner1_persptransform[:,0,1]))
    cor2_miny = np.float64(np.min(corners_2[:,0,1]))
    #print ('cor1 m in = ' +str(cor1_minx))
    
    if (cor1_minx < cor2_minx):
        minx = np.float64(cor1_minx)
    else:
        minx = np.float64(cor2_minx) 
    
    if (cor1_miny < cor2_miny):
        miny = np.float64(cor1_miny)
    else:
        miny = np.float64(cor2_miny) 
        
    cor1_maxx = np.float64(np.max(corner1_persptransform[:,0,0]))
    cor2_maxx = np.float64(np.max(corners_2[:,0,0]))
    cor1_maxy = np.float64(np.max(corner1_persptransform[:,0,1]))
    cor2_maxy = np.float64(np.max(corners_2[:,0,1]))

    if (cor1_maxx < cor2_maxx):
        maxx = np.float64(cor2_maxx)
    else:
        maxx = np.float64(cor1_maxx) 
    
    if (cor1_maxy < cor2_maxy):
        maxy = np.float64(cor2_maxy)
    else:
        maxy = np.float64(cor1_maxy) 
        
    
    min[0] = np.float64(minx)
    min[1] = np.float64(miny)
    max[0] = np.float64(maxx)
    max[1] = np.float64(maxy)
    """
    min[0] = np.float64(-1090.02)
    min[1] = np.float64(-498.366)
    max[0] = np.float64(3200)
    max[1] = np.float64(2389.4819)
    """
    #print('min = ' + str(min))
    #print('max = ' + str(max))
    
    return (min,max)


def warpCanvas(image, homography, min_xy, max_xy):
    """Warps the input image according to the homography transform and embeds
    the result into a canvas large enough to fit the next adjacent image
    prior to blending/stitching.

    Follow these steps:

        1. Create a translation matrix (numpy.ndarray) that will shift
           the image by x_min and y_min. This looks like this:

            [[1, 0, -x_min],
             [0, 1, -y_min],
             [0, 0, 1]]

        2. Compute the dot product of your translation matrix and the
           homography in order to obtain the homography matrix with a
           translation.

        NOTE: Matrix multiplication (dot product) is not the same thing
              as the * operator (which performs element-wise multiplication).
              See Numpy documentation for details.

        3. Call cv2.warpPerspective() and pass in image 1, the combined
           translation/homography transform matrix, and a vector describing
           the dimensions of a canvas that will fit both images.

        NOTE: cv2.warpPerspective() is touchy about the type of the output
              shape argument, which should be an integer.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale or color image (test cases only use uint8 channels)

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between two sequential
        images in a panorama sequence

    min_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the top left corner of a
        canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    max_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the bottom right corner of
        a canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    Returns
    -------
    numpy.ndarray(dtype=image.dtype)
        An array containing the warped input image embedded in a canvas
        large enough to join with the next image in the panorama; the output
        type should match the input type (following the convention of
        cv2.warpPerspective)

    Notes
    -----
        (1) You must explain the reason for multiplying x_min and y_min
        by negative 1 in your writeup.
    """
    # canvas_size properly encodes the size parameter for cv2.warpPerspective,
    # which requires a tuple of ints to specify size, or else it may throw
    # a warning/error, or fail silently
    canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))
    # WRITE YOUR CODE HERE
    #print ('canvas_size =', canvas_size)
    translation_mat = np.float32(np.identity(3))
    translation_mat[0,2] = np.float32(-min_xy[0])
    translation_mat[1,2] = np.float32(-min_xy[1])
    #print translation_mat
    new_mat = np.dot(translation_mat,homography)
    
    new_image = cv2.warpPerspective(image, new_mat, (canvas_size))
    #new_image = cv2.warpPerspective(image, new_mat, (canvas_size))
    #print new_mat
    #cv2.imshow('image_new.png', new_image)
    #cv2.waitKey(0)
    #print new_image
    return new_image


def blendImagePair(image_1, image_2, num_matches):
    """This function takes two images as input and fits them onto a single
    canvas by performing a homography warp on image_1 so that the keypoints
    in image_1 aligns with the matched keypoints in image_2.

    **************************************************************************

        You MUST replace the basic insertion blend provided here to earn
                         credit for this function.

       The most common implementation is to use alpha blending to take the
       average between the images for the pixels that overlap, but you are
                    encouraged to use other approaches.

           Be creative -- good blending is the primary way to earn
                  Above & Beyond credit on this assignment.

    **************************************************************************

    Parameters
    ----------
    image_1 : numpy.ndarray
        A grayscale or color image

    image_2 : numpy.ndarray
        A grayscale or color image

    num_matches : int
        The number of keypoint matches to find between the input images

    Returns:
    ----------
    numpy.ndarray
        An array containing both input images on a single canvas

    Notes
    -----
        (1) This function is not graded by the autograder. It will be scored
        manually by the TAs.

        (2) The inputs may be either color or grayscale, but they will never be
        mixed; both images will either be color, or both will be grayscale.

        (3) You can modify this function however you see fit -- e.g., change
        input parameters, return values, etc. -- to develop your blending
        process.
    """
    
    #cv2.imshow('image1',image_1)
    #cv2.waitKey(0)
    
    #cv2.imshow('image1',image_2)
    #cv2.waitKey(0)
    
    
    kp1, kp2, matches = findMatchesBetweenImages(
        image_1, image_2, num_matches)
    homography = findHomography(kp1, kp2, matches)
    #print image_1.shape
    
    corners_1 = getImageCorners(image_1)
    corners_2 = getImageCorners(image_2)
    min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)
    #print min_xy
    #print max_xy
    #print (max_xy - min_xy)
    output_image = warpCanvas(image_1, homography, min_xy, max_xy)
    #print len(output_image)
    #print len(output_image[0])
    # WRITE YOUR CODE HERE - REPLACE THIS WITH YOUR BLENDING CODE
    min_xy = min_xy.astype(np.int)
    rows = len(output_image)
    cols = len(output_image[0])
    
    image2_cols = image_2.shape[1]
    max_col = np.uint32(image2_cols/3)
    weight_mat = np.zeros((1,max_col), dtype = np.float64)
    
    #step = (1 / image2_cols)
    #print ('step' + str(step))
    weight = 0
    step = np.divide(1.0,np.float(max_col))
    for i in range(0,max_col):
        weight_mat[0,i] =  weight
        weight = weight + step
    
    #print weight_mat
    m_2 = -min_xy[1]
    for m in range(0, image_2.shape[0]):
        n_2 = -min_xy[0]
        for n in range(0, image_2.shape[1]):
            
            #if ((output_image[m_2][n_2][0] == 255) and (output_image[m_2][n_2][1] == 255) and (output_image[m_2][n_2][2] == 255)):
            #if (n > (image_2.shape[1])/80):
            if (n >= max_col):
                output_image[m_2][n_2][0] = (image_2[m][n][0])
                output_image[m_2][n_2][1] = (image_2[m][n][1])
                output_image[m_2][n_2][2] = (image_2[m][n][2])            
            
            else:
                #print('n =' +str(n))
                #print ('m=' +str(m))
                output_image[m_2][n_2][0] = np.uint32((weight_mat[0,n] * image_2[m][n][0] + (1- weight_mat[0,n]) * output_image[m_2][n_2][0]))
                output_image[m_2][n_2][1] = np.uint32((weight_mat[0,n] * image_2[m][n][1] + (1- weight_mat[0,n]) * output_image[m_2][n_2][1]))
                output_image[m_2][n_2][2] = np.uint32((weight_mat[0,n] * image_2[m][n][2] + (1- weight_mat[0,n]) * output_image[m_2][n_2][2]))
                 
            
              
                #print ('hi')

            n_2 = n_2 + 1
            #if (m_2 >= 2832):
                #print ('n_2 = ' + str(n_2))
        
        m_2 = m_2 + 1
        #print ('m = ' + str(m_2))
                
    #print image_2.shape[1]
    #print image_2.shape[0]
    #print min_xy[0]
    #print min_xy[1]
    '''
    output_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],
                 -min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2
    '''
    return output_image
    # END OF FUNCTION
