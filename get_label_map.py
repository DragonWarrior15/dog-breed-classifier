import os
import json

label_map = {}
for folder in os.listdir(os.path.join('data', 'dogImages', 'train')):
    label_map[int(folder[:3]) - 1] = folder[4:].replace('_', ' ').title()

with open('label_map', 'w') as f:
    json.dump(label_map, f)
