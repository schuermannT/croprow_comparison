import os
from pathlib import Path
from random import randint
import cv2 as cv


image_set_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'datasets', 'rosario', 'sequence01', 'zed')
ready_to_label_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'datasets', 'ma_dataset', 'rosario')
pre_mask_path = os.path.join(ready_to_label_path, 'pre_masks')
finished_imgs_path = os.path.join(ready_to_label_path, 'imgs')
  
#woodrat video 4: 320 von 530 done
#rosario seq 1: 400 done

class PreMask():
  def __init__(self):
    self.imgs = list()
    self.filenames = list()
    self.pt_color = [0, 0, 255]
    if not os.path.exists(pre_mask_path):
        Path(pre_mask_path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(finished_imgs_path):
        Path(finished_imgs_path).mkdir(parents=True, exist_ok=True)
    pass
    
  def load_images_from_directory(self, dirpath:str):
    try:
      for filename in os.listdir(dirpath):
        _img = cv.imread(os.path.join(dirpath, filename))
        if _img is not None:
          self.imgs.append(cv.resize(_img, (1280, 720)))
          self.filenames.append(filename)
    except:
      print('Problem with loading the images') 

  def open_mask_file(self):
    mask_filename = self.filename[:-4] + '.csv'
    filepath = os.path.join(pre_mask_path, mask_filename)
    self.mask_file = open(file=filepath, mode='w')

  def run(self):
    img_counter = 0
    for _img, self.filename in zip(self.imgs, self.filenames):
      img_counter = img_counter + 1
      if img_counter > 400 and img_counter <= 400:
        self.img = _img.copy()
        self.open_mask_file()
        print('showing image ' + str(img_counter) + ' of ' + str(len(self.imgs)))
        cv.imshow(self.filename, self.img)
        cv.setMouseCallback(self.filename, self.click_event)
        cv.waitKey(0)
        cv.destroyWindow(self.filename)
        finished_filename = os.path.join(finished_imgs_path, self.filename)
        cv.imwrite(finished_filename, _img)
        self.mask_file.close()

  def click_event(self, event, x, y, flags, params):
    if event != 0:
      if event == cv.EVENT_LBUTTONDOWN:
        coord_str = str(x) + ', ' + str(y)
        self.mask_file.write(coord_str + '\n')
        cv.circle(self.img, (x,y), 0, self.pt_color, 5)
        cv.imshow(self.filename, self.img)
      elif event == cv.EVENT_MBUTTONDOWN:
        coord_str = str(-1) + ', ' + str(-1)
        self.mask_file.write(coord_str + '\n')
        self.pt_color[0] = randint(0, 256)
        self.pt_color[1] = randint(0, 256)
        self.pt_color[2] = randint(0, 256)
      
  
if __name__=='__main__':
  pre_masker = PreMask()
  pre_masker.load_images_from_directory(image_set_path)
  pre_masker.run()
