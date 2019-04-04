"""
Adapted from the MonoHand3D codebase for the MonocularRGB_3D_Handpose project (github release)

support methods used in the hand pose pipeline
@author: Paschalis Panteleris (padeler@ics.forth.gr)
"""

import PyMBVCore as Core
import PyMBVDecoding as Decondign
import PyMBVAcquisition 
import PyMBVRendering as Rendering
import PyMBVLibraries as Libraries
import numpy as np
import cv2


class HandVisualizer(object):
    def __init__(self, mmanager, viz_dims=(1600, 900)):
        self.mmanager = mmanager
        w, h = viz_dims
        renderer = Rendering.RendererOGLCudaExposed.get(w,h)
        renderer.culling = Rendering.RendererOGLBase.Culling.CullNone
        erenderer = Rendering.ExposedRenderer(renderer, renderer)
        rhelper = Libraries.RenderingHelper()
        rhelper.renderer = renderer
        raccess = Libraries.RenderingAccessor()
        raccess.exposer = erenderer

        self.rhelper = rhelper
        self.raccess = raccess
        self.renderer = renderer

    def render(self, hand_model, pose, clb, flag=Rendering.Renderer.WriteFlag.WriteAll):

        self.rhelper.bonesMap = hand_model.bones_map
        self.rhelper.decoder = hand_model.decoder

        self.renderer.setSize(1, 1, int(clb.width), int(clb.height))
        self.renderer.uploadViewMatrices(Core.MatrixVector([clb.camera.Graphics_getViewTransform()]))
        self.renderer.uploadProjectionMatrices(Core.MatrixVector([clb.camera.Graphics_getProjectionTransform()]))

        self.rhelper.render(flag, 1, 1, int(clb.width), int(clb.height), [pose, ], self.mmanager)

    def getDepth(self, color_map = cv2.COLORMAP_RAINBOW):
        depth = self.raccess.getPositionMap()[:, :, 2]
        mask = np.ones(depth.shape, dtype=np.ubyte)
        mask[depth == 0] = 0
        dmap = cv2.normalize(depth,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1, mask)

        if color_map is not None:
            dmapColor = cv2.applyColorMap(dmap, cv2.COLORMAP_RAINBOW)
            dmapColor[dmap == 0] = 0
            return dmapColor
        else:
            return dmap

    def showDepth(self, windowTitle="Depth"):
        dmapColor = self.getDepth()
        cv2.imshow(windowTitle, dmapColor)

    def showNormals(self, windowTitle="Normals"):
        nmap = (self.raccess.getNormalMap()[:, :, :3] * 255).astype(np.ubyte)
        cv2.imshow(windowTitle, nmap)




def draw_rect(viz, label, bb, box_color=(155, 100, 100), text_color=(220, 200, 200)):
    x, y, w, h = bb
    cv2.rectangle(viz, (x, y), (x + w, y + h), box_color, 2)
    cv2.putText(viz, label, (x+4, y+h-10), 0, 0.5, text_color, 1, cv2.LINE_AA)

def draw_hands(viz, bb):
    left = bb[:4]
    right = bb[4:]
    if left[2] > 0:
        draw_rect(viz, "", left, box_color=(0, 255, 0), text_color=(200, 200, 0))
    if right[2] > 0:
        draw_rect(viz, "", right)


def draw_points2D(img, points, color, size, show_idx=False):

    for idx, p in enumerate(points):
        center = tuple(int(v) for v in p)
        cv2.circle(img, center, size, color)
        if show_idx:
            cv2.putText(img, str(idx), center, 0, 0.3, (71, 99, 255), 1)

    return img


