"""
This is based on the vot_wrapper.py included with D3S and adapted to work with
the VOT2020 dataset to test with segmentation mask predictions instead of rotated
bounding boxes as was done in VOT2018.

This is adapted from the python_ncc_mask.py VOT integration example file linked below.

https://github.com/votchallenge/integration/blob/master/python/python_ncc_mask.py
"""

import vot
import sys
import cv2
import os
import numpy as np
import collections
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add pytracking package to system path to enable loading of dependencies
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.tracker.segm import Segm
from pytracking.parameter.segm import default_params as vot_params


def rect_to_poly(rect):
    x0 = rect[0]
    y0 = rect[1]
    x1 = rect[0] + rect[2]
    y1 = rect[1]
    x2 = rect[0] + rect[2]
    y2 = rect[1] + rect[3]
    x3 = rect[0]
    y3 = rect[1] + rect[3]
    return [x0, y0, x1, y1, x2, y2, x3, y3]

def parse_sequence_name(image_path):
    tmp_path = image_path.replace('\\','/') 
    idx = tmp_path.find('/color/')
    return tmp_path[idx - tmp_path[:idx][::-1].find('/'):idx], idx

def parse_frame_name(image_path, idx):
    tmp_path = image_path.replace('\\','/') 
    frame_name = tmp_path[idx + len('/color/'):]
    return frame_name[:frame_name.find('.')]

def rect_from_mask(mask):
    '''
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    '''
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

def mask_from_rect(rect, output_sz):
    '''
    create a binary mask from a given rectangle
    rect: axis-aligned rectangle [x0, y0, width, height]
    output_sz: size of the output [width, height]
    '''
    mask = np.zeros((output_sz[1], output_sz[0]), dtype=np.uint8)
    x0 = max(int(round(rect[0])), 0)
    y0 = max(int(round(rect[1])), 0)
    x1 = min(int(round(rect[0] + rect[2])), output_sz[0])
    y1 = min(int(round(rect[1] + rect[3])), output_sz[1])
    mask[y0:y1, x0:x1] = 1
    return mask

# MAIN

# Evaluation mode. If rectangle is selected it will return rectangular bounding boxes.
# Polygon will return the locations of 4 corners of a polygon (rotated rectangle).
# Mask will return a segmentation mask. Mask is new to VOT2020.
mode = "mask" # | "rectangle", "mask", or "polygon"

handle = vot.VOT(mode)
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

params = vot_params.parameters()

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

if mode == "polygon":
    gt_rect = [round(selection.points[0].x, 2), round(selection.points[0].y, 2),
            round(selection.points[1].x, 2), round(selection.points[1].y, 2),
            round(selection.points[2].x, 2), round(selection.points[2].y, 2),
            round(selection.points[3].x, 2), round(selection.points[3].y, 2)]
elif mode == "rectangle":
    gt_bbox = [selection.x, selection.y, selection.width, selection.height]
elif mode == "mask":
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    mask = make_full_size(selection, (image.shape[1], image.shape[0]))
    gt_bbox = rect_from_mask(mask)

sequence_name, idx_ = parse_sequence_name(imagefile)
frame_name = parse_frame_name(imagefile, idx_)

params.masks_save_path = ''
params.save_mask = False
params.return_mask = True # Added parameter for VOT2020

tracker = Segm(params)

# tell the sequence name to the tracker (to save segmentation masks to the disk)
tracker.sequence_name = sequence_name
tracker.frame_name = frame_name

if mode == "polygon":
    tracker.initialize(image, gt_rect)
elif mode == "rectangle":
    tracker.initialize(image, gt_bbox)
elif mode == "mask":
    tracker.initialize(image, gt_bbox, init_mask=mask)

# If true, this will start saving prediction masks to the disk at the location specified below
debug_mask = False

if debug_mask and mode == "mask":
    # Create new directory for mask output
    mask_folder = os.path.join(os.path.dirname(__file__),'..','..','masks2020')
    if not os.path.exists(mask_folder):
        print('Making directory:',mask_folder)
        os.mkdir(mask_folder)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

    # tell the frame name to the tracker (to save segmentation masks to the disk)
    frame_name = parse_frame_name(imagefile, idx_)
    tracker.frame_name = frame_name

    prediction = tracker.track(image)

    if mode == "polygon" and len(prediction) == 4:
        prediction = rect_to_poly(prediction)

    if mode == "polygon":
        pred_poly = vot.Polygon([vot.Point(prediction[0], prediction[1]),
                                vot.Point(prediction[2], prediction[3]),
                                vot.Point(prediction[4], prediction[5]),
                                vot.Point(prediction[6], prediction[7])])
        handle.report(pred_poly)
    elif mode == "rectangle":
        pred_bbox = vot.Rectangle(*prediction)
        handle.report(pred_bbox)
    elif mode == "mask":
        # Retrieve the mask from the tracker
        pred_mask = tracker.get_result_mask()

        # Below creates a mask from the bbox prediction for debugging
        # pred_mask = mask_from_rect(prediction, (image.shape[1], image.shape[0]))

        if debug_mask:
            # This will save the mask to the disk
            mask_save_dir = os.path.join(mask_folder, sequence_name)
            if not os.path.exists(mask_save_dir):
                os.mkdir(mask_save_dir)
            mask_save_path = os.path.join(mask_save_dir, '%s.png' % frame_name)
            cv2.imwrite(mask_save_path, pred_mask)

        handle.report(pred_mask)
