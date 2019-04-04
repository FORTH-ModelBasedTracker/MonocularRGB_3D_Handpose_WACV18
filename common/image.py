"""
Adapted from PyCvUtils project for the MonocularRGB_3D_Handpose project.

@author: Paschalis Panteleris (padeler@ics.forth.gr)
"""

import numpy as np
import cv2


def show(label, img, delay=-1):
    if len(img.shape)==2 and img.dtype!=np.uint8:
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
    cv2.imshow(label,img)
    
    if delay>-1:
        return cv2.waitKey(delay)


