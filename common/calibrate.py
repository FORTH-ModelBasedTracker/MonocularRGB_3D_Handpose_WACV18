"""
Adapted from PyCvUtils project for the MonocularRGB_3D_Handpose project.

@author: Paschalis Panteleris (padeler@ics.forth.gr)
"""

import json
import numpy as np
import PyMBVCore as Core

def PackOpenCVCalib(repErr,mtx, dist, rvec, tvec,imgSize,cameraId=0):

    camera = {
        "Dims":imgSize,
        "CameraMatrix":mtx.tolist(),
        "Distortion":dist.flatten().tolist(),
        "Rotation":rvec.flatten().tolist(),
        "Translation":tvec.flatten().tolist(),
        "CalibReprError":repErr,
        "CameraID":cameraId,
    }
    return camera


def OpenCVCalib2CameraMeta(calibDict,znear=100,zfar=10000):
    cameraFrustrum = Core.CameraFrustum()
    mtx = np.array(calibDict["CameraMatrix"],dtype=np.float32)
    size = tuple(calibDict["Dims"])
    k1,k2,p1,p2,k3 = calibDict["Distortion"]
    cameraId = calibDict["CameraID"]
    rvec = np.array(calibDict["Rotation"],dtype=np.float32)
    tvec = np.array(calibDict["Translation"],dtype=np.float32)

    cameraFrustrum.OpenCV_setIntrinsics(mtx,size,znear,zfar)
    cameraFrustrum.OpenCV_setExtrinsics(tvec,rvec)
    cameraMeta = Core.CameraMeta(cameraFrustrum,size[0],size[1],k1,k2,p1,p2,k3,cameraId)

    return cameraMeta

def CameraMeta2OpenCVCalib(cameraMeta):
    dims = cameraMeta.size
    dist = np.array([cameraMeta.k1,cameraMeta.k2,cameraMeta.p1,cameraMeta.p2,cameraMeta.k3],dtype=np.float32)
    mtx = cameraMeta.camera.OpenCV_getIntrinsics(dims)
    tvec,rvec = cameraMeta.camera.OpenCV_getExtrinsics()
    camId = cameraMeta.cameraId
    return PackOpenCVCalib(-1,mtx,dist,rvec,tvec,dims,camId)


def SaveOpenCVCalib(calibDict, filename):
    with open(filename,"w") as f:
        json.dump(calibDict,f, indent=4, separators=(',', ': '))


def LoadOpenCVCalib(filename):
    with open(filename,"r") as f:
        calibDict = json.load(f)
        return calibDict



