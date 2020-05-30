import numpy as np
import pandas as pd
import uuid
from PIL import Image
import torch
from torch.autograd import Variable

from models import fnn_model
from models import cnn_pytorch


class Models():
    def __init__(self):
        self.params = np.load('models/updated_weights.npy', allow_pickle=True)[()]
        self.cnn_path = 'models/cnn_mnist.pt'

    def get_image(self, img):

        filename = 'digit' + '__' + str(uuid.uuid1()) + '.jpg'

        with open('tmp/' + filename, 'wb') as f:
            f.write(img)

        img = Image.open('tmp/' + filename).convert('L')
        bbox = Image.eval(img, lambda px: 255 - px).getbbox()

        if bbox == None:
            return None
        widthlen = bbox[2] - bbox[0]
        heightlen = bbox[3] - bbox[1]

        if heightlen > widthlen:
            widthlen = int(20.0 * widthlen / heightlen)
            heightlen = 20
        else:
            heightlen = int(20.0 * widthlen / heightlen)
            widthlen = 20

        hstart = int((28 - heightlen) / 2)
        wstart = int((28 - widthlen) / 2)

        img_temp = img.crop(bbox).resize((widthlen, heightlen), Image.NEAREST)

        new_img = Image.new('L', (28, 28), 255)
        new_img.paste(img_temp, (wstart, hstart))

        imgdata = list(new_img.getdata())

        img_array = np.array([(255.0 - x) / 255.0 for x in imgdata])

        img_cnn_np = 255.0 - np.array(imgdata)
        img_cnn_array = torch.from_numpy(img_cnn_np).type(torch.LongTensor)
        img_cnn_array = img_cnn_array.view(-1, 1, 28, 28).float() / 255.0

        return img_array, img_cnn_array

    def cnn_make_prediction(self):
        pass

    def train_models(self):
        pass

    def predict_label(self, img):

        img_array, img_cnn_array = self.get_image(img)

        if img_array is None:
            return "Can't recognize - nothing is drawn!"

        fnn = fnn_model.TwoLayerNet(self.params, mode='predict')
        pred_label = fnn.predict(img_array)

        ##############
        cnn = cnn_pytorch.CNN()
        cnn.load_state_dict(torch.load(self.cnn_path))
        cnn.eval()

        cnn_pred_proba, cnn_pred_label = torch.topk(cnn(img_cnn_array), 10)
        cnn_pred_proba_np = cnn_pred_proba.detach().numpy().flatten()
        cnn_pred_label_np = cnn_pred_label.detach().numpy().flatten()
        #print(cnn_pred_proba_np)

        ##############

        ans_dict = {'fnn1': str(pred_label[0][0]),
                    'fnn1_proba': str(pred_label[0][1])+'%',
                    'fnn2': str(pred_label[1][0]),
                    'fnn2_proba': str(pred_label[1][1])+'%',
                    'fnn3': str(pred_label[2][0]),
                    'fnn3_proba': str(pred_label[2][1])+'%',

                    'cnn1': str(cnn_pred_label_np[0]),
                    'cnn1_proba': str(np.round(float(cnn_pred_proba_np[0]), 4)*100) + '%',
                    'cnn2': str(cnn_pred_label_np[1]),
                    'cnn2_proba': str(np.round(float(cnn_pred_proba_np[1]), 4)*100) + '%',
                    'cnn3': str(cnn_pred_label_np[2]),
                    'cnn3_proba': str(np.round(float(cnn_pred_proba_np[2]), 4)*100) + '%'
            }
        return ans_dict
