'''
Adapted from the MonoHand3D codebase for the MonocularRGB_3D_Handpose project (github release)

@author: Paschalis Panteleris (padeler@ics.forth.gr)
'''

import numpy as np
import cv2
import tensorflow as tf


# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17], # palm
           [1, 2], [2, 3], [3,4], # thump
           [5, 6], [6, 7], [7, 8], # index
           [9, 10], [10, 11], [11, 12], # middle
           [13, 14], [14, 15], [15, 16], # ring
           [17, 18], [18, 19], [19, 20], # pinky
        ]

# visualize
colors = [[255,255,255], 
          [255, 0, 0], [255, 60, 0], [255, 120, 0], [255, 180, 0],
          [0, 255, 0], [60, 255, 0], [120, 255, 0], [180, 255, 0],
          [0, 255, 0], [0, 255, 60], [0, 255, 120], [0, 255, 180],
          [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],
          [0, 0, 255], [60, 0, 255], [120, 0, 255], [180, 0, 255],]

def peaks_to_hand(peaks, dx,dy):
    hand = []
    for joints in peaks:
        sel = sorted(joints, key=lambda x: x.score, reverse=True)
        
        if len(sel)>0:
            p = sel[0]
            x,y,score = p.x+dx, p.y+dy, p.score
            hand.append([x,y,score])
        else:
            hand.append([0,0,0])
        
    return np.array(hand,dtype=np.float32)


def visualize_2dhand_skeleton(canvas, hand, stickwidth = 3, thre=0.1):
  
    for pair in limbSeq:
        if hand[pair[0]][2]>thre and hand[pair[1]][2]>thre:
            x0,y0 = hand[pair[0]][:2]
            x1,y1 = hand[pair[1]][:2]
            x0 = int(x0)
            x1 = int(x1)
            y0 = int(y0)
            y1 = int(y1)    
            cv2.line(canvas,(x0,y0), (x1,y1), colors[pair[1]%len(colors)], thickness=stickwidth,lineType=cv2.LINE_AA)

    return canvas

def visualize_3dhand_skeleton(canvas, hand, stickwidth = 10, txt_size=0.5):
  
    for pair in limbSeq:
            x0,y0 = hand[pair[0]][:2]
            x1,y1 = hand[pair[1]][:2]
            x0 = int(x0)
            x1 = int(x1)
            y0 = int(y0)
            y1 = int(y1)    
            cv2.line(canvas,(x0,y0), (x1,y1), colors[pair[1]%len(colors)], thickness=stickwidth,lineType=cv2.LINE_AA)

    for i, p in enumerate(hand):
        x,y = int(p[0]), int(p[1])
        cv2.circle(canvas, (x,y), 4, colors[i%len(colors)], thickness=stickwidth, lineType=cv2.LINE_AA)
        if txt_size>0:
            cv2.putText(canvas, str(i), (x+5,y), 0, txt_size, colors[i%len(colors)],lineType=cv2.LINE_AA)


    return canvas


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def preprocess(oriImg, boxsize=368, stride=8, padValue=128):
    scale = float(boxsize) / float(oriImg.shape[0])

    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

    return input_img, pad


def update_bbox(p2d, dims, pad=0.3):
    x = np.min(p2d[:,0])
    y = np.min(p2d[:,1])
    xm = np.max(p2d[:,0])
    ym = np.max(p2d[:,1])

    cx = (x+xm)/2
    cy = (y+ym)/2
    w = xm - x
    h = ym - y
    b = max((w,h,224))
    b = int(b + b*pad)

    x = cx-b/2
    y = cy-b/2

    x = max(0,int(x))
    y = max(0,int(y))

    x = min(x, dims[0]-b)
    y = min(y, dims[1]-b)
    

    return [x,y,b,b]



class Estimator(object):
    def __init__(self, model_file, input_layer="input_1", output_layer="k2tfout_0"):
        
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer

        self.graph = load_graph(model_file)
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)
        self.sess = tf.Session(graph=self.graph)
        

    def predict(self, img):
        results = self.sess.run(self.output_operation.outputs[0], feed_dict={self.input_operation.outputs[0]: img})
        return np.squeeze(results)

