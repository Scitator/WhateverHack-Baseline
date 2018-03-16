from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn


class BaselineHeadModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_size = kwargs.get("left_size") + kwargs.get("right_size")
        self.net = nn.Sequential(OrderedDict([
            ("bn_0", nn.BatchNorm1d(input_size)),
            ("linear_1", nn.Linear(in_features=input_size, out_features=256)),
            ("bn_1", nn.BatchNorm1d(256)),
            ("act_1", nn.ReLU()),
            ("linear_2", nn.Linear(in_features=256, out_features=128)),
            ("bn_2", nn.BatchNorm1d(128)),
            ("act_2", nn.ReLU()),
            ("logits", nn.Linear(in_features=128, out_features=1)),
            ("probability", nn.Sigmoid())
        ]))

    def forward(self, x):
        x = torch.cat((x['left'], x['right']), 1)
        x = self.net.forward(x)
        return x


class BaselineModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.image_net = kwargs.get("image_net")
        self.text_net = kwargs.get("text_net")
        for model in [self.image_net, self.text_net]:
            for param in model.parameters():
                param.requires_grad = False
        self.head_net = BaselineHeadModel(**kwargs)

    def forward(self, x):
        image_enc = self.image_net.forward(x["image"])
        text_enc = self.text_net.forward(x["text"])
        # @TODO: pytorch bug? why double instead of float?
        text_enc = text_enc.type(torch.cuda.FloatTensor) \
            if torch.cuda.is_available() \
            else text_enc.type(torch.FloatTensor)
        middle = {"left": image_enc, "right": text_enc}
        result = self.head_net.forward(middle)
        return result


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def make_vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ReduceMean(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        x = torch.mean(x, dim=self.dim)
        return x


def build_baseline_model(**kwargs):
    base = [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 'C',
        512, 512, 512, 'M',
        512, 512, 512]

    image_net = nn.Sequential(*(make_vgg(base, 3) + [nn.AdaptiveMaxPool2d((1,1)), Flatten()]))
    vgg_weights = torch.load(kwargs.get("vgg_weights"))
    image_net.load_state_dict(vgg_weights)

    embeddings = np.load(kwargs.get("embeddings"))
    embeddings = torch.from_numpy(embeddings)
    embedding = nn.Embedding(embeddings.size(0), embeddings.size(1))
    embedding.weight = nn.Parameter(embeddings)
    text_net = nn.Sequential(embedding, ReduceMean(1))

    kwargs["left_size"] = 1024
    kwargs["right_size"] = embeddings.size(1)

    net = BaselineModel(
        image_net=image_net,
        text_net=text_net,
        **kwargs)

    return net
