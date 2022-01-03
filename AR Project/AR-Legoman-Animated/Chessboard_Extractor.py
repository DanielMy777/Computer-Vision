import cv2
import glob, os

files = glob.glob('./media/chessboard/*.jpeg')
for file in files:
    os.remove(file)

cap = cv2.VideoCapture('./media/chessboard/vid-chessboard.mp4')
if (cap.isOpened() == False):
  print("Error opening video file")


img_num = 1
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret:
    if(img_num % 50 == 0):
        cv2.imwrite(f"./media/chessboard/image-chessboard-{int(img_num/50)}.jpeg", frame)
  else:
    break
  img_num+=1
 

#print("done")
cap.release()
cv2.destroyAllWindows()