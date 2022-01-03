# %% [markdown]
# # Project 2 extra: AR - Custom 3D Model
# # Daniel Malky 318570462
# # Amit Viner 208538892

# ======= imports
import numpy as np
import cv2
from glob import glob
import pickle
import os.path
from Camera_Calibrator import CameraCalibrator
from mesh_renderer_custom import MeshRenderer

# ======= constants
DO_NEW_CALIBRATION = False
calibration_log_file = 'Calibration_Log.pk'
TH = 0.915
TH_ALL = 1
pixel_to_cm = 0.0264583333
square_size = 2.9
chess_shape = (7,4)
feature_extractor = cv2.SIFT_create()
bf = cv2.BFMatcher()
real_box_height = 27.5      # inaccurate sizes in order to control scale of model
real_box_width = 18.5

lego_path = "./media/objs/Legoman/legoman.obj"

# === template image keypoint and descriptors
template_im = cv2.imread('./media/template.png', cv2.IMREAD_UNCHANGED)
template_rgb = cv2.cvtColor(template_im, cv2.COLOR_BGR2RGB)
template_gray = cv2.cvtColor(template_im, cv2.COLOR_BGR2GRAY)
kp_template, desc_template = feature_extractor.detectAndCompute(template_gray, None)

im_box_height = template_im.shape[0] * pixel_to_cm
im_box_width = template_im.shape[1] * pixel_to_cm
world_im_ratio_h = real_box_height / im_box_height
world_im_ratio_w = real_box_width / im_box_width


# ===== video input, output and metadata
cap = cv2.VideoCapture('./media/input.mp4')
if (cap.isOpened() == False):
  print("Error opening video file")

orig_frame_width = frame_width = int(cap.get(3))
orig_frame_height = frame_height = int(cap.get(4))

out = cv2.VideoWriter('output-legoman.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (orig_frame_width,orig_frame_height))

# ===== Camera Calibration
if(DO_NEW_CALIBRATION is False and os.path.exists(calibration_log_file) is True):
  with open(calibration_log_file, 'rb') as cal:
      (camera_k, camera_dist_coefs) = pickle.load(cal)
else:
  import Chessboard_Extractor #Activates chessboard extractor program
  img_mask = "./media/chessboard/*.jpeg"
  img_names = glob(img_mask)
  num_images = len(img_names)

  cali = CameraCalibrator(square_size, chess_shape)
  for i, fn in enumerate(img_names):
      cali.load_image(fn, True)

  camera_k = cali.get_camera_K_n_dist_coefs()[0]
  camera_dist_coefs = cali.get_camera_K_n_dist_coefs()[1]

#======= Renderer
lego_renderer = MeshRenderer(camera_k, orig_frame_width, orig_frame_height, lego_path)

# ======= Process a single frame
def process_frame(frame):
    # ====== find keypoints matches of frame and template
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = feature_extractor.detectAndCompute(frame_gray, None)

    matches = bf.knnMatch(desc_template, desc_frame, k=2)

    good_and_second_good_match_list = []
    for m in matches:
      if m[0].distance/m[1].distance < TH:
        good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]

    # ======== find homography
    good_kp_template = np.array([kp_template[m.queryIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)

    # ======== take subset of keypoints that obey homography (both frame and reference)
    inlier_kp_template = np.array([kp_template[m.queryIdx].pt for i, m in enumerate(good_match_arr) if masked[i][0] != 0])
    inlier_kp_frame = np.array([kp_frame[m.trainIdx].pt for i, m in enumerate(good_match_arr) if masked[i][0] != 0])

    # ======== solve PnP to get cam pose (r_vec and t_vec)
    x_kp_template = np.array([[kp[0]] for kp in inlier_kp_template])
    y_kp_template = np.array([[kp[1]] for kp in inlier_kp_template])
    x_kp_template = x_kp_template * world_im_ratio_w * pixel_to_cm
    y_kp_template = y_kp_template * world_im_ratio_h * pixel_to_cm
    real_world_kp_template = np.array([np.append([x], np.append([y], [0])) for (x, y) in zip(x_kp_template, y_kp_template)])

    ret, r_vec, t_vec = cv2.solvePnP(real_world_kp_template, inlier_kp_frame, camera_k, camera_dist_coefs)
    
    result = frame_bgr.copy()
    
    lego_renderer.draw(result, r_vec, t_vec)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result


# ==== Main
# ========== run on all frames
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    result = process_frame(frame)
  else:
    break

  out.write(result)
  cv2.imshow('out', result)

  if cv2.waitKey(10) & 0xFF == ord('q'):
      break


cap.release()
out.release()
cv2.destroyAllWindows()