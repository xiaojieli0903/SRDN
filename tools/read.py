import cv2   
import sys
sys.path.append('/opt/ros/hydro/lib/python2.7/dist-packages')
img = cv2.imread("./output.png")   
cv2.namedWindow("Image")   
cv2.imshow("Image", img)   
cv2.waitKey (0)  
cv2.destroyAllWindows() 
