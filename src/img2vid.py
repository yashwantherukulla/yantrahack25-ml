import cv2
import glob

image_files = sorted(glob.glob("splits_final_deblurred/train/data/*.PNG"))[:100]

frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
video = cv2.VideoWriter('train_100.mp4', fourcc, 2, (width, height))

for img in image_files:
    frame = cv2.imread(img)
    video.write(frame)

video.release()
cv2.destroyAllWindows()
