# %% [markdown]
# # Project 1: Lane Detection
# # Daniel Malky 318570462
# # Amit Viner 208538892

import numpy as np
import matplotlib.pyplot as plt
import cv2

# %%
# ## Import a video
cv2.namedWindow('out',cv2.WINDOW_NORMAL)
cv2.resizeWindow('out', 800,600)

cap = cv2.VideoCapture('input.mp4')
if (cap.isOpened() == False):
  print("Error opening video file")

orig_frame_width = frame_width = int(cap.get(3))
orig_frame_height = frame_height = int(cap.get(4))

# %%
# ## Video Attributes
fig = (10,10)                           # figure size
color = (70, 170, 0)                    # line color
thick = 4                               # line thickness

# ## Set attributes for relavent zone to mask (rectangle points)
W_start = 500    
H_start = 680                      
W_end = 900
H_end = 980

# ## Additional Attributes
H_limit_up = H_start                    # starting and ending boundries for lines to draw
H_limit_down = H_end - 20
r_step = 1
t_step = np.pi / 180
TH = 30

# %%
# ## Load images that will be presented when changing lanes
left_image = cv2.imread('left.jpg', cv2.IMREAD_UNCHANGED)
left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
left_image[left_image > 240] = 0

right_image = cv2.imread('right.jpg', cv2.IMREAD_UNCHANGED)
right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
right_image[right_image > 240] = 0

max_message_dur = 100 # duration for the image to show
left_image_counter = [0]
right_image_counter = [0]

# %%
# ## Class - MyLine
# ## A class for calculating one "smart line" given a series of spreaded lines.
class MyLine:
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    Counter = 0

    # initialization.
    def __init__(self, side):  # letf == 0 right == 1
       self.side = side
    
    # add a line to the smart line calculation.
    def Add_Line(self, x1, y1, x2, y2):
        self.x1 += x1
        self.x2 += x2
        self.y1 += y1
        self.y2 += y2
        self.Counter += 1

    # calculate the smart line as the average of the accumelated lines.
    def Calc_Lane_Line(self):
        if self.Counter == 0:
            return 0
        self.x1 = int(self.x1 / self.Counter)
        self.x2 = int(self.x2 / self.Counter)
        self.y1 = int(self.y1 / self.Counter)
        self.y2 = int(self.y2 / self.Counter)
        return 1

    # returns the smart line's incline.
    def My_M(self):
        return Get_M(self.x1, self.x2, self.y1, self.y2)

    # returns whether the lines added are valid for calculation.
    def Is_Valid(self):
        return not((self.x1 == 0) and (self.x2 == 0) and (self.y1 == 0) and (self.y2 == 0))

    # adjusts the height and length of the lines to fit the frame.
    def Fix_Me(self, H_up, H_down):
        M = Get_M(self.x1, self.x2, self.y1, self.y2)
        B = self.y1 - M * self.x1
        self.y1 = H_down
        self.y2 = H_up
        self.x1 = int((self.y1 - B) / M)
        self.x2 = int((self.y2 - B) / M)


# %%
# ## Function that masks relevant zone for lane detection in a frame.
# ## @param img the source image to mask
# ## @param P_start (x,y) of the rectangle mask [top left point]
# ## @param P_end (x,y) of the rectangle mask [bottom right point]
# ## @return the masked image, everything outside the rectangle zone gets black
def Get_Mask(img, P_start,P_end):
    mask_res = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(mask_res, P_start,P_end , 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask_res)
    return masked

# %%
# ## Function that masks gets a line as two points from rho ad theta representation.
# ## @param r_t rho and theta representation of the line
# ## @return (x1, y1, x2, y2) that represent the line as two points
def Get_Linear(r_t):
    rho = r_t[0, 0]
    theta = r_t[0, 1]

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 900 * (-b))
    y1 = int(y0 + 900 * (a))
    x2 = int(x0 - 900 * (-b))
    y2 = int(y0 - 900 * (a))

    return (x1, y1, x2, y2)

# %%
# ## Function that calculates incline of a line given two points of it.
# ## @param x1, x2, y1, y2 - coordinates for the two points on the line
# ## @return desired incline
def Get_M(x1, x2, y1, y2):
    if x1 == x2:
        return 0
    return (y1 - y2) / (x1 - x2)

# %%
# ## Function that validates whether the line detected is a lane.
# ## @param x1, x2, y1, y2 - coordinates for the two points on the line
# ## @return True iff the line has a change of being our lane.
def Is_Correct_Line(x1, x2, y1, y2):
    M = Get_M(x1, x2, y1, y2)
    Res =  False
    if ( 0.9 < M and M < 3):
        Res = True
    elif (-3 < M and M < -0.9):
        Res = True
    return Res

# %%
# ## Function that identifies left lane line from right.
# ## @param x1, x2, y1, y2 - coordinates for the two points on the line
# ## @return True iff the line is a right lane line.
def Check_Right(x1, x2, y1, y2):
    return Get_M(x1, x2, y1, y2) > 0

# %%
# ## Function that detects lane changes.
# ## @param Left_line the detected left lane smart line
# ## @param Right_line the detected right lane smart line
# ## @param mid_src the x coordinate of the middle pixel in the source frame
# ## @return 'left' / 'right' / 'none' according to what lane change was found
def Detect_Lane_Change(Left_line, Right_line, mid_src):
    left_m = Left_line.My_M()
    right_m = Right_line.My_M()
    left_dist = abs((Left_line.x1/2 + Left_line.x2/2) - mid_src)
    right_dist = abs((Right_line.x1/2 + Right_line.x2/2) - mid_src)
    if(right_m > 2 and right_dist < 50):
        return 'right'
    elif(left_m < -2 and left_dist < 50):
        return 'left'
    else:
        return 'none'

# %%
# ## Function that calculates the weight of the fade thats required to the given time
# ## @param max_time is the total time the fade is supposed to take place
# ## @param curr_time is the total time the fade is already taking place
# ## @param sigma is is an adjustment parameter 
# ## @return fade weight as spoken
def Get_Faded_Weight(max_time, curr_time, sigma):
    mid = max_time / 2
    delta = None
    if(curr_time > mid):
        delta = curr_time
    else:
        delta = (max_time - curr_time)
    return delta*sigma

# %%
# ## Function that displays a message when a lane change is detected (and for further frames)
# ## @param src is the source image
# ## @param side 'left' / 'right' / 'none' according to what lane change was detected
# ## @param left_counter is the amount of frames left to present a previous left lane change
# ## @param right_counter is the amount of frames left to present a previous right lane change
# ## @return the source image with the message on top
def Handle_Lane_Change(src, side, left_counter, right_counter):
    if(left_counter[0] != 0):
        alpha = Get_Faded_Weight(max_message_dur, left_counter[0], 0.01)
        beta = 1- alpha
        src = cv2.addWeighted(src, alpha, left_image, beta, 0)
        left_counter[0] -= 1
    elif(right_counter[0] != 0):
        alpha = Get_Faded_Weight(max_message_dur, right_counter[0], 0.01)
        beta = 1- alpha
        src = cv2.addWeighted(src, alpha, right_image, beta, 0)
        right_counter[0] -= 1
    elif(side == 'left'):
        left_counter[0] = 100
    elif(side == 'right'):
        right_counter[0] = 100
    
    return src

# %%
# ## Function that processes and draws lane lines on a frame.
# ## @param image the source image
# ## @return image after drawing and marking
def process_image(image):
    dst = image.copy()
    Im_gary = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    (thr, Gray_thr_img) = cv2.threshold(Im_gary, 127, 255, cv2.THRESH_BINARY)
    Gaus_img = cv2.GaussianBlur(Gray_thr_img, (7, 7), 0)  # make less noise
    Canny_Img = cv2.Canny(Gaus_img, 50, 170)
    Mask_img = Get_Mask(Canny_Img,(W_start,H_start),(W_end,H_end))

    lines = cv2.HoughLines(Mask_img, r_step, t_step, TH)
    Left_Line = MyLine(0)
    Right_Line = MyLine(1)

    # no lines detected
    if lines is None:
        return image

    for r_t in lines:
        (x1, y1, x2, y2) = Get_Linear(r_t)
        if Is_Correct_Line(x1, x2, y1, y2):    #dicide whether the line represents a lane or a mistake
            if not Check_Right(x1, x2, y1, y2):
                Left_Line.Add_Line(x1, y1, x2, y2)
            else:
                Right_Line.Add_Line(x1, y1, x2, y2)

    Left_Line.Calc_Lane_Line()
    Right_Line.Calc_Lane_Line()

    if Left_Line.Is_Valid():
      Left_Line.Fix_Me(H_limit_up, H_limit_down)
      dst = cv2.line(dst, (Left_Line.x1, Left_Line.y1), (Left_Line.x2, Left_Line.y2), color,
                       thickness=thick)

    if Right_Line.Is_Valid():
       Right_Line.Fix_Me(H_limit_up, H_limit_down)
       dst = cv2.line(dst, (Right_Line.x1, Right_Line.y1), (Right_Line.x2, Right_Line.y2), color,
                         thickness=thick)

    lane_change = Detect_Lane_Change(Left_Line, Right_Line, int(orig_frame_width/2))
    dst = Handle_Lane_Change(dst, lane_change, left_image_counter, right_image_counter)
    lane_change = 'none'

    return dst


# %%
# ## -Main code snippet-
# ## Read each frame of the source video, process it, and paste it on the output video.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (orig_frame_width,orig_frame_height))
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        reframe = process_image(frame)
    else:
        break

    out.write(reframe)
    cv2.imshow('out', reframe)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



cap.release()
out.release()
cv2.destroyAllWindows()

# %%
