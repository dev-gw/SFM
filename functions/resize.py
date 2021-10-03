import os
import glob
from PIL import Image

files = glob.glob('C:/Users/smartlab/Desktop/Color ball/*.jpg') # 이미지 파일 경로

i = 55 # 파일 이름 시작번호
for f in files:
    img = Image.open(f)
    img_resize = img.resize((416,416)) # 픽셀 사이즈
    img_resize.save('C:/Users/smartlab/Desktop/GBC_balls/export/images/' + str(i) + '.jpg') # 저장 경로
    i += 1