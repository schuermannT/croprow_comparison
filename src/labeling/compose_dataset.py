import os
import shutil
from tqdm import tqdm
from random import shuffle

destination = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'datasets', 'ma_dataset', 'combined')
source = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'datasets', 'ma_dataset')
set_names = {'crdld_test', 'crdld_train', 'rosario', 'woodrat'}
sets = {'crdld':   [], 
        'rosario': [], 
        'woodrat': []}

print("Extracting filenames")
for set in set_names:
  if set.__contains__('crdld'):
    for filename in tqdm(os.listdir(os.path.join(source, set, 'img_labels')), desc=set):
      sets['crdld'].append(filename)
  elif set.__contains__('rosario'):
    for filename in tqdm(os.listdir(os.path.join(source, set, 'img_labels')), desc=set):
      sets['rosario'].append(filename)
  elif set.__contains__('woodrat'):
    for filename in tqdm(os.listdir(os.path.join(source, set, 'img_labels')), desc=set):
      sets['woodrat'].append(filename)
  else:
    print(f"ERROR: can't find set {set}")

print("Splitting filenames into train, val, test by ration 8:1:1")
train_set = []
val_set = []
test_set = []

for set in sets.values():
  small_set_size = round(0.1 * len(set))
  shuffle(set)
  for i in range(small_set_size):
    val_set.append(set.pop())
    test_set.append(set.pop())
  train_set = train_set + set

print(f"train: {len(train_set)} val: {len(val_set)} test: {len(test_set)}")
  
for v_file in tqdm(val_set, desc='val'):
  for set in set_names:
    check_mask_file = os.path.join(source, set, 'img_labels', v_file)
    check_img_file = os.path.join(source, set, 'imgs', v_file)
    if os.path.isfile(check_mask_file) and os.path.isfile(check_img_file):
      shutil.copy(check_mask_file, os.path.join(destination, 'val', 'masks', v_file))
      shutil.copy(check_img_file, os.path.join(destination, 'val', 'imgs', v_file))
for t_file in tqdm(test_set, desc='test'):
  for set in set_names:
    check_mask_file = os.path.join(source, set, 'img_labels', t_file)
    check_img_file = os.path.join(source, set, 'imgs', t_file)
    if os.path.isfile(check_mask_file) and os.path.isfile(check_img_file):
      shutil.copy(check_mask_file, os.path.join(destination, 'test', 'masks', t_file))
      shutil.copy(check_img_file, os.path.join(destination, 'test', 'imgs', t_file))
for tr_file in tqdm(train_set, desc='train'):
  for set in set_names:
    check_mask_file = os.path.join(source, set, 'img_labels', tr_file)
    check_img_file = os.path.join(source, set, 'imgs', tr_file)
    if os.path.isfile(check_mask_file) and os.path.isfile(check_img_file):
      shutil.copy(check_mask_file, os.path.join(destination, 'train', 'masks', tr_file))
      shutil.copy(check_img_file, os.path.join(destination, 'train', 'imgs', tr_file))

print('Finished composing dataset')