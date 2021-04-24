"""
This file is intended to produce annotated images and videos from the output of 
the VOT python toolkit evaluation. This will annotate the sequence images with
the predicted masks or bounding boxes. It can also annotate the images with the 
sequence and tracker information and a legend and will then produce a .mp4 video 
from the image frames for all 60 sequences. By default, the annotation text is
disabled.

Expected Folder Structure:
-D3S
    -ltr
    -pytracking
        -vot_mask_vis.py
-vot-2020
    -sequences
    -results
        -DS3Python
            -Baseline
            -visualization (Where output will be saved)

This is adapted from a visualization script made for Assignment 2 of SYDE 673
by Curtis Stewart.
"""

# Import dependencies
import os
import sys
import argparse
import numpy as np  # Array Manipulation library
import matplotlib.pyplot as plt # library which generates figures
import cv2 # OpenCV library for computer vision tasks
from glob import glob # for searching for files
import os # for file reading
import pathlib # for more directory management
import sys # for manipulating the cell output
from tqdm import tqdm, trange # for progress bar
import re # for reformatting strings
from statistics import mean

def main():
    parser = argparse.ArgumentParser(description='Produce VOT visualization')
    parser.add_argument('--dataset', type=str, default='vot-2020', help='Name of dataset folder. vot-2020 or vot-2018')
    parser.add_argument('--tracker', type=str, default='D3SPython', help='Name of result folder.')
    parser.add_argument('--output', type=str, default='Visualization', help='Name of output folder.')
    parser.add_argument('--mode', type=str, default='mask', help='bbox or mask')
    parser.add_argument('--annotate', type=int, default=0, help='Annotation level: ' + 
        '0 - tracking prediction, 1 - with legend, 2 - with seq info')

    args = parser.parse_args()

    if args.mode == "bbox":
        add_gt_mask = False
        add_pred_mask = False
        add_gt_bbox = True
        add_pred_bbox = True
    elif args.mode == "mask":
        add_gt_mask = False
        add_pred_mask = True
        add_gt_bbox = True
        add_pred_bbox = True

    # Check current directory
    cwd_head, cur_folder = os.path.split(os.getcwd())

    if cur_folder == 'D3S':
        data_path = os.path.join(os.getcwd(),'..',args.dataset)
    elif cur_folder == 'pytracking':
        data_path = os.path.join(os.getcwd(),'..','..',args.dataset)
    else:
        print('Unknown root directory. Please run this from either the D3S or pytracking folder')
        return

    data_path = os.path.normpath(data_path)
    print('data_path:', data_path)

    vis_folder = os.path.join(data_path,'results',args.tracker,'visualization')
    result_folder = os.path.join(data_path,'results',args.tracker,'baseline')

    # Create new directory for visualization output
    if not os.path.exists(vis_folder):
        print('Creating vis_folder:',vis_folder)
        os.mkdir(vis_folder)
    else:
        print('Using vis_folder:',vis_folder)

    # Search through sequence folders
    sequences = glob(os.path.join(data_path,'sequences','*','color'))
    num_seq = len(sequences)

    print(f"Found {num_seq} sequences")

    # Create FPS file
    fps_path = os.path.join(vis_folder,'FPS.txt')
    fps_file = open(fps_path,"w")

    total_fps_list = []

    # Create a progress bar | ncols=500
    p_bar = tqdm(sequences, total=num_seq, position=0, leave=True, unit='seq')

    # Loop through all test sequences
    for seq_path in p_bar:
        # replace any backslashes with forward slash (windows compatibility)
        seq_path = seq_path.replace('\\','/') 
        seq = seq_path.split('/')[-2]
        p_bar.set_description('Seq {0}'.format(seq))
        p_bar.update()

        # Create output folder
        out_folder = os.path.join(vis_folder,seq)
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        # Load ground truth bboxes
        gt_file = os.path.join(data_path,'sequences',seq,'groundtruth.txt')
        gt_reader = open(gt_file,'r')
        gt_line = gt_reader.readline()
        if len(gt_line) > 0 and gt_line[0] == 'm':
            gt_state = gt_line[1:].split(',')
        else:
            gt_state = gt_line.split(',')

        if len(gt_state) != 4 and len(gt_state) != 8:
            gt_state = gt_state[:4]

        if len(gt_line[1:].split(',')) > 5:
            gt_raw_mask = gt_line[1:].split(',')
        else:
            gt_raw_mask = None

        # Load result bbox output file
        if args.dataset == "vot-2020":
            result_file = os.path.join(result_folder,seq,seq + '_00000000.txt')
        elif args.dataset == "vot-2018":
            result_file = os.path.join(result_folder,seq,seq + '_001.txt')
        else:
            result_file = os.path.join(result_folder,seq,seq + '_00000000.txt')
        result_reader = open(result_file,'r')
        line = result_reader.readline()
        if len(line) > 0 and line[0] == 'm':
            state = line[1:].split(',')
        else:
            state = line.split(',')
        if len(state) != 4 and len(state) != 8:
            state = state[:4]

        if len(line) > 0 and len(line[1:].split(',')) > 5:
            pred_raw_mask = line[1:].split(',')
        else:
            pred_raw_mask = None

        # Load time output file
        if args.dataset == "vot-2020":
            time_file = os.path.join(result_folder,seq,seq + '_00000000_time.value')
        elif args.dataset == "vot-2018":
            time_file = os.path.join(result_folder,seq,seq + '_001_time.value')
        else:
            time_file = os.path.join(result_folder,seq,seq + '_00000000_time.value')
        time_reader = open(time_file,'r')
        ftime = float(time_reader.readline())

        fps_list = []

        # Loop through images in sequence folder
        seq_imgs = sorted(glob(seq_path+'/*'))
        for img_i, img_file in enumerate(seq_imgs, start=1):
            # replace any backslashes with forward slash (windows compatibility)
            img_file = img_file.replace('\\','/')
            img_path, img_name = os.path.split(img_file)

            # Load sequence frame
            frame = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

            if add_gt_mask and gt_raw_mask is not None:
                # Add ground truth mask
                mask_color = (0, 255, 0) # BGR format
                frame = add_raw_mask(frame, gt_raw_mask,frame.shape[1],frame.shape[0],mask_color)

            if add_gt_bbox and len(gt_state) == 4:
                # Plot gt bbox as green rectangle
                (x, y, w, h) = [int(float(v)) for v in gt_state]
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif add_gt_bbox and len(gt_state) == 8:
                # Plot gt polygon as green rectangle
                gt_state = list(map(float, gt_state)) # convert str to float
                pts = np.array(gt_state, dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(frame,[pts],True,(0, 255, 0), 2)

            if add_pred_mask and pred_raw_mask is not None:
                # Add prediction mask as yellow mask
                mask_color = (0, 255, 255) # BGR format
                frame = add_raw_mask(frame, pred_raw_mask,frame.shape[1],frame.shape[0],mask_color)

            if add_pred_bbox and len(state) == 4:
                # Plot tracker bbox as red rectangle
                (x, y, w, h) = [int(float(v)) for v in state]
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            elif add_pred_bbox and len(state) == 8:
                # Plot tracker polygon as red rectangle
                state = list(map(float, state)) # convert str to float
                pts = np.array(state, dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(frame,[pts],True,(0, 0, 255), 2)

            # Calculate fps from frame time
            if ftime != 0:
                fps_list.append(1/ftime)
                total_fps_list.append(1/ftime)
                fps = "{:.2f}".format(1/ftime)
            else:
                fps = "N/A"

            if args.annotate > 1:
                # Add text annotations
                if args.dataset == "vot-2020":
                    data_name = "VOT2020"
                elif args.dataset == "vot-2018":
                    data_name = "VOT2018"
                else:
                    data_name = args.dataset
                info = [f"Sequence: {data_name}/{seq}",
                        f"Tracker: {args.tracker}",
                        f"Tracker FPS: {fps}"]

                # text properties
                scale = frame.shape[0]/720*1
                font = cv2.FONT_HERSHEY_SIMPLEX
                thick = min(int(scale),1)

                # Get size of text field
                xt, yt, wt, ht = int(scale*5), int(scale*5), 0, int(scale*20)
                for text in info:
                    (label_width, label_height), baseline = cv2.getTextSize(text, font, scale, thick)
                    wt = max(label_width,wt)
                    ht += label_height + baseline
                wt += int(scale*10)

                if seq == 'flamingo1':
                    # Set info to bottom left of frame
                    yt = max(0,frame.shape[0] - ht - int(scale*5))

                # Add black rectangular overlay
                sub_img = frame[yt:yt+ht, xt:xt+wt]
                blk_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                res = cv2.addWeighted(sub_img, 0.5, blk_rect, 0.5, 0)
                frame[yt:yt+ht, xt:xt+wt] = res
                
                # loop over the info tuples and draw them on the frame
                for (i, text) in enumerate(info):
                    cv2.putText(frame, text, (xt + int(scale*5), \
                        yt + int((i+1)*(label_height + baseline) + scale*5)), \
                        font, scale, (255, 255, 255), thick)
            
            if args.annotate > 0:
                # Add legend
                legend = ["Legend","Green: GT","Red: Tracker"]

                # Get size of legend
                wt, ht = 0, int(scale*20)
                for text in legend:
                    (label_width, label_height), baseline = cv2.getTextSize(text, font, scale, thick)
                    wt = max(label_width,wt)
                    ht += label_height + baseline
                wt += int(scale*10)
                xt = max(0,frame.shape[1] - wt - int(scale*5))
                yt = max(0,frame.shape[0] - ht - int(scale*5))

                # Add black rectangular overlay for legend
                sub_img = frame[yt:yt+ht, xt:xt+wt]
                blk_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                res = cv2.addWeighted(sub_img, 0.5, blk_rect, 0.5, 0)
                frame[yt:yt+ht, xt:xt+wt] = res

                # Legend text colors
                lcolor = [(255, 255, 255),(150, 255, 150), (150, 150, 255)]
                
                # loop over the legend tuples and draw them on the frame
                for (i, text) in enumerate(legend):
                    cv2.putText(frame, text, (xt + int(scale*5), \
                        yt + int((i+1)*(label_height + baseline) + scale*5)), \
                        font, scale, lcolor[i], thick)

            # Save resulting image file
            outfile = os.path.join(out_folder,img_name)
            cv2.imwrite(outfile,frame)

            # Create video from images
            if img_i <= 1:
                frameSize = (frame.shape[1], frame.shape[0])
                outvid_file = os.path.join(out_folder,seq+'.mp4')
                # video will play at 20 fps
                out = cv2.VideoWriter(outvid_file,cv2.VideoWriter_fourcc(*'mp4v'), 20, frameSize)

            # Write image to video
            out.write(frame)

            # Go to next line in files
            line = result_reader.readline()
            # tracker bbox
            if len(line) > 0 and line[0] == 'm':
                state = line[1:].split(',')
            else:
                state = line.split(',')
            if len(state) != 4 and len(state) != 8:
                state = state[:4]
            # tracker mask
            if len(line) > 0 and len(line[1:].split(',')) > 5:
                pred_raw_mask = line[1:].split(',')
            else:
                pred_raw_mask = None
            # Ground truth data
            gt_line = gt_reader.readline()
            if len(gt_line) > 0 and gt_line[0] == 'm':
                gt_state = gt_line[1:].split(',')
            else:
                gt_state = gt_line.split(',')
            if len(gt_state) != 4 and len(gt_state) != 8:
                gt_state = gt_state[:4]
            if len(gt_line[1:].split(',')) > 5:
                gt_raw_mask = gt_line[1:].split(',')
            else:
                gt_raw_mask = None
            # time information
            time_line = time_reader.readline().strip("\n")
            if len(time_line) > 0:
                ftime = float(time_line) # frame time
            
        # Close files
        result_reader.close()
        time_reader.close()
        gt_reader.close()

        # Release video writer
        out.release()

        # Write mean sequence fps to file
        fps_file.write("{}: {:f}\n".format(seq,mean(fps_list)))
        
        # break

    # Write average fps over dataset to file
    fps_file.write("Total FPS: {:f}\n".format(mean(total_fps_list)))
    fps_file.write("Min FPS: {:f}\n".format(min(total_fps_list)))
    fps_file.write("Max FPS: {:f}\n".format(max(total_fps_list)))
    fps_file.close()


def add_raw_mask(frame, raw_mask, im_width, im_height, mask_color):
    # Convert mask to integers
    raw_mask = list(map(int, raw_mask))

    # Retrieve bounding box information
    # print('bbox:',raw_mask[:4])
    (x, y, w, h) = raw_mask[:4]
    raw_mask = raw_mask[4:]

    # create sub_mask size of bounding box
    sub_mask = np.zeros((h,w), dtype=np.uint8)
    mask_border = np.zeros((h,w), dtype=np.uint8)

    # Flatten mask
    sub_mask = sub_mask.flatten()
    mask_border = mask_border.flatten()

    if len(sub_mask) != sum(raw_mask):
        print("Error in mask format!!")

    idx = 0

    # This will convert the mask information in the prediction result file from
    # running vot evaluate to a mask array.
    # sub_mask will contain the actual mask, and mask_border contains the border
    # of the mask.
    for i in range(len(raw_mask)//2):
        idx += raw_mask[2*i]
        if (2*i + 1) < len(raw_mask):
            sub_mask[idx:idx + raw_mask[2*i+1]] = 1
            mask_border[idx] = 1
            idx += raw_mask[2*i+1]
            mask_border[idx-1] = 1

    # Reshape mask
    sub_mask = sub_mask.reshape((h,w))
    mask_border = mask_border.reshape((h,w))

    # Create colored mask
    # color_mask = np.zeros((h,w,3), dtype=np.uint8)
    color_mask = np.copy(frame[y:y+h, x:x+w])
    color_mask[np.nonzero(sub_mask)] = mask_color

    # Blend mask with image
    sub_img = frame[y:y+h, x:x+w]
    res = cv2.addWeighted(sub_img, 0.5, color_mask, 0.5, 0)
    # Color mask border as solid color
    res[np.nonzero(mask_border)] = mask_color
    frame[y:y+h, x:x+w] = res

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # fig = plt.figure(figsize=(8, 4.5))
    # plt.imshow(frame)
    # plt.show()

    return frame


if __name__ == '__main__':
    main()
