import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

class CameraCalibrator:
    def __init__(self, square_size, pattern_size):
        self.square_size = square_size
        self.pattern_size = pattern_size
        self.pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        self.pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        self.pattern_points *= square_size

        self.reset_all()

    def reset_all(self):
        self.image_points = []
        self.object_points = []
        self.im_height = None
        self.im_width = None
        
    def load_image(self, image_dest, with_print):
        imgBGR = cv2.imread(image_dest)
        if imgBGR is None:
            print("ERROR - Failed to load", image_dest)
        else:
            imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
            if self.im_height == None and self.im_width == None:
                self.im_height = img.shape[0]
                self.im_width = img.shape[1]
            
            if not(self.im_width == img.shape[1] and self.im_height == img.shape[0]):
                print("ERROR - Unmatching image sizes", image_dest)
                return
            
            found, corners = cv2.findChessboardCorners(img, self.pattern_size)
            if not found:
                print("ERROR - Chessboard not found")
                return

            if with_print:
                img_w_corners = cv2.drawChessboardCorners(imgRGB, self.pattern_size, corners, found)
                plt.subplot(222)
                plt.imshow(img_w_corners)
            
            self.image_points.append(corners.reshape(-1, 2))
            self.object_points.append(self.pattern_points)

    def get_camera_K_n_dist_coefs(self):
        rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(self.object_points, self.image_points, (self.im_width, self.im_height), None, None)

        with open('Calibration_Log.pk', 'wb') as cal:
            pickle.dump((camera_matrix, dist_coefs), cal)

        return (camera_matrix, dist_coefs)