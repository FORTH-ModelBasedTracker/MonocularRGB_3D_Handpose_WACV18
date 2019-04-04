"""
Adapted from PyCvUtils project for the MonocularRGB_3D_Handpose project.

@author: Paschalis Panteleris (padeler@ics.forth.gr)
"""

import cv2
from common.calibrate import OpenCVCalib2CameraMeta, LoadOpenCVCalib

class OpenCVGrabber(object):
    '''
    A wrapper grabber for the opencv VideoCapture object that exposes the MBV Grabber API
    '''


    def __init__(self, cam_id = 0, calib_file = None, mirror = False):
        self.cam_id = cam_id
        self.mirror = mirror
        if calib_file is not None:
            self.calib = OpenCVCalib2CameraMeta(LoadOpenCVCalib(calib_file))

        else:
            self.calib = None

        self.cap = cv2.VideoCapture(cam_id)
        # TODO Get these values from parameters in the constructor
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.8)

    def initialize(self):
        pass

    def grab(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("VideoCapture.read() returned False")
            if self.mirror:
                frame = cv2.flip(frame, 1)

            return [frame,], [self.calib,]
        except Exception as e:
            print("Failed to grab image from opencv capture object. %s" % e)
            raise e



if __name__ == '__main__':
    import sys
    src = 0
    if len(sys.argv)>1:
        src = sys.argv[1]
        try:
            src = int(src)
        except: # ignore if not an int (probably a url of file path)
            pass
        print("Grabbing from ", src)
    grabber  = OpenCVGrabber(src)
    k = 0 
    while k & 0xFF!=ord('q'):
        imgs, clbs = grabber.grab()
        cv2.imshow("GRAB", imgs[0])
        k = cv2.waitKey(1)
