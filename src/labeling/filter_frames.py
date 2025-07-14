import os
from pathlib import Path
import cv2 as cv

frames_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'datasets', 'WoodratTradeCo', 'test_video', 'video_5')
  
class FrameFilter():
  def __init__(self):
    self.imgs = list()
    self.filenames = list()
    self.removed_counter = 0
    self.marked_for_del = list()
    pass
    
  def load_images_from_directory(self, dirpath:str):
    try:
      for filename in os.listdir(dirpath):
        _img = cv.imread(os.path.join(dirpath, filename))
        if _img is not None:
          self.imgs.append(_img)
          self.filenames.append(filename)
    except:
      print('Problem with loading the images')

  def run(self):
    img_counter = 0
    for _img, self.filename in zip(self.imgs, self.filenames):
      img_counter = img_counter + 1
      print('showing image ' + str(img_counter) + ' of ' + str(len(self.imgs)))
      cv.imshow(self.filename, _img)
      cv.setMouseCallback(self.filename, self.click_event)
      cv.waitKey(0)
      if cv.getWindowProperty(self.filename, cv.WND_PROP_VISIBLE) > 0:
        cv.destroyWindow(self.filename)
    for filename in self.marked_for_del:
      del_img_path = os.path.join(frames_path, filename)
      if os.path.exists(del_img_path):
        os.remove(del_img_path)
        self.removed_counter = self.removed_counter + 1
      else:
        print('could not remove unedited image ' + filename)
    print('deleted ' + str(self.removed_counter) + ' images of ' + str(len(self.imgs)))


  def click_event(self, event, x, y, flags, params):
    if event != 0:
      if event == cv.EVENT_LBUTTONDOWN:
        self.marked_for_del.append(self.filename)
        print(self.filename + ' marked for deletion')
      if event == cv.EVENT_MBUTTONDOWN:
        if self.marked_for_del.__contains__(self.filename):
          self.marked_for_del.pop()
          print(self.filename + ' not marked for deletion')
      
  
if __name__=='__main__':
  pre_masker = FrameFilter()
  pre_masker.load_images_from_directory(frames_path)
  pre_masker.run()
