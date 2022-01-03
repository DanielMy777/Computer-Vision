# %% [markdown]
# # Project 2: Prespective Warping
# # Daniel Malky 318570462
# # Amit Viner 208538892

# ======= imports
import numpy as np
import cv2

# ======= constants
TH = 0.915
TH_ALL = 1
NOISE_TH = 45
feature_extractor = cv2.SIFT_create()
bf = cv2.BFMatcher()

# === template image keypoint and descriptors
template_im = cv2.imread('./media/template.png', cv2.IMREAD_UNCHANGED)
template_rgb = cv2.cvtColor(template_im, cv2.COLOR_BGR2RGB)
template_gray = cv2.cvtColor(template_im, cv2.COLOR_BGR2GRAY)
kp_template, desc_template = feature_extractor.detectAndCompute(template_gray, None)

#---for debug---
#test = cv2.drawKeypoints(template_im, kp_template, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('test', test)
#---for debug---

# ===== video input, output and metadata
cap = cv2.VideoCapture('./media/input.mp4')
if (cap.isOpened() == False):
  print("Error opening video file")

orig_frame_width = frame_width = int(cap.get(3))
orig_frame_height = frame_height = int(cap.get(4))

out = cv2.VideoWriter('output-pres-warp.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (orig_frame_width,orig_frame_height))

spidey = cv2.imread('./media/spidey.jpg', cv2.IMREAD_UNCHANGED)
spidey_rgb = cv2.cvtColor(spidey, cv2.COLOR_BGR2RGB)
spidey_resized = cv2.resize(spidey_rgb, (template_im.shape[1], template_im.shape[0]))
spidey = spidey_resized

old_H = None

# ===== Enhancement: confirm the H matrix is valid and not just noise
def confirm_H(old_H, new_H):
  #diff = ab = np.sum(abs(old_H - new_H))          <<< Remove comments to activate enhancement
  #if(diff > NOISE_TH):
    #return old_H
  return new_H

# ===== processing a single frame
def process_frame(frame):
    global old_H
    # ====== find keypoints matches of frame and template
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = feature_extractor.detectAndCompute(frame_gray, None)

    matches = bf.knnMatch(desc_template, desc_frame, k=2)

    good_and_second_good_match_list = []
    for m in matches:
      if m[0].distance/m[1].distance < TH:
        good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]

    # --- for debug ---
    #im_matches = cv2.drawMatchesKnn(template_rgb, kp_template, frame_rgb, kp_frame,
    #                           good_and_second_good_match_list[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv2.imshow('test', im_matches)
    # --- for debug ---

    # ======== find homography
    good_kp_template = np.array([kp_template[m.queryIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr])
    ransac_thr = len(good_match_arr) / 30
    H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, ransac_thr)
    
    if(old_H is not None):
      H = confirm_H(old_H, H)
    old_H = H

    # ======== do warping of chechik image on template image
    spidey_warped = cv2.warpPerspective(spidey, H, (frame_rgb.shape[1], frame_rgb.shape[0]))
    spidey_mask = (spidey_warped == 0)
    result = spidey_warped
    result[spidey_mask]= frame_rgb[spidey_mask]

    return result


# ===== Main
# ========== run on all frames
while(cap.isOpened()):
  ret, frame = cap.read()
  
  if ret == True:
    result = process_frame(frame)
  else:
    break

  result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
  out.write(result)
  cv2.imshow('out', result)

  if cv2.waitKey(10) & 0xFF == ord('q'):
      break


cap.release()
out.release()
cv2.destroyAllWindows()

