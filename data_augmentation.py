import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
def load_data(path):
  train_x = sorted(glob(os.path.join(path, "Train", "Images", "*.jpg")))
  train_y = sorted(glob(os.path.join(path, "Train", "Masks", "*.jpg")))
  test_x = sorted(glob(os.path.join(path, "Test", "Images", "*.jpg")))
  test_y = sorted(glob(os.path.join(path, "Test", "Masks", "*.jpg")))
  return (train_x, train_y), (test_x, test_y)
def create_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)
def augment_data(Images, Masks, save_path, augment =True):
  size = (512, 512)
  for idx, (x,y) in tqdm(enumerate(zip(Images, Masks)), total = len(Images)):
    name = x.split("/")[-1].split(".")[0] 
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    y = cv2.imread(y, cv2.IMREAD_COLOR)   
    if augment == True:
      aug = HorizontalFlip(p=1.0)
      augmented = aug(image=x, mask=y)
      x1= augmented["image"]
      y1= augmented["mask"]
      aug = VerticalFlip(p=1.0)
      augmented = aug(image=x, mask=y)
      x2= augmented["image"]
      y2= augmented["mask"]
      aug = Rotate(limit=45, p=1.0)
      augmented = aug(image=x, mask=y)
      x3= augmented["image"]
      y3= augmented["mask"]
      X = [x, x1, x2, x3]
      Y = [y, y1, y2, y3]
    else:
      X = [x]
      Y = [y]
    index = 0 
    for i, m in zip(X, Y):
      i = cv2.resize(i, size)
      m = cv2.resize(m, size)
      tmp_image_name = f"{name}_{index}.jpg"
      tmp_mask_name = f"{name}_{index}.jpg"
      image_path = os.path.join (save_path, "Images", tmp_image_name)
      mask_path = os.path.join (save_path, "Masks", tmp_mask_name)
      cv2.imwrite(image_path, i)
      cv2.imwrite(mask_path, m)
      index += 1
if __name__ == "__main__":
  np.random.seed(42)
  data_path = "/data/thussain/"
  (train_x, train_y), (test_x, test_y) = load_data(data_path)
  print(f"Train: {len(train_x)} . {len(train_y)}")
  print(f"Test: {len(test_x)} . {len(test_y)}")
  create_dir("/data/thussain/new_data/Train/Images/")
  create_dir("/data/thussain/new_data/Train/Masks/")
  create_dir("/data/thussain/new_data/Test/Images/")
  create_dir("/data/thussain/new_data/Test/Masks/")
  augment_data(train_x, train_y, "data/thussain/new_data/Train/", augment = False)
  augment_data(test_x, test_y, "data/thussain/new_data/Test/", augment = False)
