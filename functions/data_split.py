from glob import glob
from sklearn.model_selection import train_test_split

img_list_1 = glob('C:/obj/*.png')
img_list_2 = glob('C:/obj/*.jpg')
img_list = img_list_1 + img_list_2

train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=1000)
print(len(train_img_list), len(val_img_list))

with open('C:/darknet-master/data/train.txt', 'w') as f:
  f.write('\n'.join(train_img_list)+'\n')

with open('C:/darknet-master/data/test.txt', 'w') as f:
  f.write('\n'.join(val_img_list)+'\n')