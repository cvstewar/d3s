import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time
import os
import numpy as np


class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params

    def initialize(self, image, state, class_info=None):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        #self.sequence_name = sequence.name
        self.initialize(image, sequence.init_state)
        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)

        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, sequence.init_state)

        # Track
        tracked_bb = [sequence.init_state]
        for frame in sequence.frames[1:]:
            image = self._read_image(frame)

            start_time = time.time()
            state = self.track(image)
            times.append(time.time() - start_time)

            if len(state) != len(sequence.init_state):
                # Note that the tracker may return a state of length 4 instead of 8 if 
                # using a rotated bounding box and the target wasn't found. In this case,
                # reuse the prediction from the last frame
                tracked_bb.append(tracked_bb[-1])
            else:
                tracked_bb.append(state)

            if self.params.visualization:
                self.visualize(image, state)

        return tracked_bb, times

    def track_videofile(self, videofilepath, optional_box=None):
        """Run track with a video file input."""

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)
        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, list, tuple)
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            self.initialize(frame, optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                self.initialize(frame, init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                return

            frame_disp = frame.copy()

            # Draw box
            state = self.track(frame)
            state = [int(s) for s in state]
            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                self.initialize(frame, init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def track_webcam(self):
        """Run tracker with webcam."""

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.mode_switch = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                    self.mode_switch = True
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'track'
                    self.mode_switch = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = ui_control.get_bb()
                self.initialize(frame, init_state)

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                state = self.track(frame)
                state = [int(s) for s in state]
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Put text
            font_color = (0, 0, 0)
            if ui_control.mode == 'init' or ui_control.mode == 'select':
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            elif ui_control.mode == 'track':
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        # self.fig.canvas.manager.window.move(800, 50)
        self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (100, 50))

        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()


    def visualize(self, image, state):
        self.ax.cla()
        self.ax.imshow(image)

        if len(state) == 4:
            pred = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=2, edgecolor='r', facecolor='none')
        elif len(state) == 8:
            p_ = np.array(state).reshape((4, 2))
            pred = patches.Polygon(p_, linewidth=2, edgecolor='r', facecolor='none')
        else:
            print('Error: Unknown prediction region format.')
            exit(-1)

        self.ax.add_patch(pred)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if self.pause_mode:
            plt.waitforbuttonpress()

    def _read_image(self, image_file: str):
        return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)

