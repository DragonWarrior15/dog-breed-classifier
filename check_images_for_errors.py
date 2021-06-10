import os
from torchvision.io import read_image

parent_dir = 'data/dogImages'

img_list = []

for train in os.listdir(parent_dir):
    for di in os.listdir(os.path.join(parent_dir, train)):
        img_class = int(di[:3]) - 1
        img_label = di[4:]
        for f in os.listdir(os.path.join(parent_dir, train, di)):
            img_list.append((img_class, img_label,
                                    os.path.join(parent_dir, train, di, f)))

for idx in range(len(img_list)):
    try:
        img = read_image(img_list[idx][2])
    except:
        print('Error at', img_list[idx][2])

# remove the image data/dogImages/train/098.Leonberger/Leonberger_06571.jpg
