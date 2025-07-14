import os
import shutil
from tqdm import tqdm
from random import shuffle

destination = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'datasets', 'ma_dataset', 'combined')
source = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'datasets', 'ma_dataset')
set_names = {'crdld_test', 'crdld_train', 'rosario', 'woodrat'}

print("Splitting filenames into train, val, test by ration 8:1:1")
train_set = []
for file in os.listdir(os.path.join(destination, 'train', 'imgs')):
  train_set.append(file[:-4]+'.csv')
val_set = []
for file in os.listdir(os.path.join(destination, 'val', 'imgs')):
  val_set.append(file[:-4]+'.csv')
test_set = []
for file in os.listdir(os.path.join(destination, 'test', 'imgs')):
  test_set.append(file[:-4]+'.csv')
  
for v_file in tqdm(val_set, desc='val'):
  for set in set_names:
    check_label_file = os.path.join(source, set, 'labels', v_file)
    if os.path.isfile(check_label_file):
      shutil.copy(check_label_file, os.path.join(destination, 'val', 'labels', v_file))
for t_file in tqdm(test_set, desc='test'):
  for set in set_names:
    check_label_file = os.path.join(source, set, 'labels', t_file)
    if os.path.isfile(check_label_file):
      shutil.copy(check_label_file, os.path.join(destination, 'test', 'labels', t_file))
for tr_file in tqdm(train_set, desc='train'):
  for set in set_names:
    check_label_file = os.path.join(source, set, 'labels', tr_file)
    if os.path.isfile(check_label_file):
      shutil.copy(check_label_file, os.path.join(destination, 'train', 'labels', tr_file))

print('Finished composing dataset')