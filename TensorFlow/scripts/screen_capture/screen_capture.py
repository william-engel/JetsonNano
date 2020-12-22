import time
import cv2
import mss
import numpy
import win32gui
import os

os.startfile('C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Moorhuhn Remake\\Moorhuhn Remake spielen.lnk')
time.sleep(15)

hwnd = win32gui.FindWindow(None, 'Moorhuhn1Remake') 

win32gui.SetForegroundWindow(hwnd)

with mss.mss() as sct:
  n = 0
  while "Screen capturing":
      last_time = time.time()
      
      try:
        bbox = win32gui.GetWindowRect(hwnd)
      except:
        quit()
      
      img = numpy.array(sct.grab(bbox))

      if n % 20 == 0:
        cv2.imwrite('Tensorflow/img/frame_%.3d.jpg' %n, img)
      
      print("fps: {}".format(1 / (time.time() - last_time)))

      n += 1

      # Press "q" to quit
      if cv2.waitKey(25) & 0xFF == ord("q"):
          cv2.destroyAllWindows()
          break