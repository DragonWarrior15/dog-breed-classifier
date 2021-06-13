# python app_server.py
import os
from flask import Flask, request, render_template, jsonify

from utils import dogBreedDataset, dogBreedTransforms, dogBreedClassifier, logger_setup

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.io import read_image

# global definitions of image transforms and the model
input_size = 32
output_size = 10
transforms = dogBreedTransforms(resize_size=input_size)
model = dogBreedClassifier(input_size=input_size, output_size=output_size)
# model is picked based on the highest test set accuracy from the logs
model.load_state_dict(torch.load(os.path.join('models', '202106121107', '00087')))
model.eval()
# softmax
sm = nn.Softmax(dim=1)
# class to labels mapping
label_map = {
    0: 'Affenpinscher',
    1: 'Afghan_hound',
    2: 'Airedale_terrier',
    3: 'Akita',
    4: 'Alaskan_malamute',
    5: 'American_eskimo_dog',
    6: 'American_foxhound',
    7: 'American_staffordshire_terrier',
    8: 'American_water_spaniel',
    9: 'Anatolian_shepherd_dog'
}
for k in label_map:
    label_map[k] = label_map[k].replace('_', ' ').title()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def start_page():
    """the app page where user can upload image"""
    return render_template('app_ui.html')

@app.route('/inference', methods=['POST'])
def inference():
    """files are stored in the request.files attr while other form entries
    are part of request.form
    https://stackoverflow.com/questions/51765498/flask-file-upload-unable-to-get-form-data
    to read the data from the passed file
    https://stackoverflow.com/questions/20015550/read-file-data-without-saving-it-in-flask
    read filestream image
    torch.tensor
    convert image to numpy array
    torch.tensor
    """
    if 'img' not in request.files:
        return jsonify({'preds': [[label_map[i], 0] for i in label_map]})
    # first save the image to disk to make a file for pytorch to read
    img = request.files['img'].read()
    with open('tmp', 'wb') as f:
        f.write(img)

    #read the image
    img = read_image('tmp').float()
    # add the batch dimension
    img = torch.unsqueeze(img, dim=0)

    # convert the image as required
    img = transforms.img_transform_test(img)

    # run the inference, apply softmax to get probabilities
    preds = sm(model(img)).tolist()[0]

    # get the top 10 probabilities in sorted order
    preds = [[i, x] for i, x in enumerate(preds)]
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    preds = preds[:10]
    preds = [[label_map[x[0]], int(100*x[1])] for x in preds]

    # convert to numpy array and return
    return jsonify({'preds': preds})


if __name__ == '__main__':
    app.run(debug = True)
