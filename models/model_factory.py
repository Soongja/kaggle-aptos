import torch.nn as nn
import pretrainedmodels
from torchvision.models import resnet18, resnet34
from efficientnet.model import EfficientNet


# ['fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d', 'inceptionv4', 'inceptionresnetv2',
# 'alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'resnet18', 'resnet34', 'resnet50',
# 'resnet101', 'resnet152', 'inceptionv3', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
# 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'nasnetamobile', 'nasnetalarge', 'dpn68', 'dpn68b', 'dpn92',
# 'dpn98', 'dpn131', 'dpn107', 'xception', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
# 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'cafferesnet101', 'pnasnet5large', 'polynet']


def get_model(config, num_classes=1):
    model_name = config.MODEL.NAME

    if model_name.startswith('resnet'):
        model = globals().get(model_name)(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name.startswith('efficient'):
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)

    else:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)

    print('model name:', model_name)

    if model_name.startswith('efficient'):
        if config.MODEL.FC_TYPE == 1:
            model.fc_type = 1
            in_features = model.out_channels
            new_fc = nn.Sequential(
                            nn.Linear(in_features, 256),
                            nn.BatchNorm1d(256, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 1))
            model._fc = new_fc
            print('new fc added')
        elif config.MODEL.FC_TYPE == 2:
            model.fc_type = 2
            in_features = model.out_channels
            new_fc = nn.Sequential(
                            nn.BatchNorm1d(in_features*2, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True),
                            nn.Dropout(0.25),
                            nn.Linear(in_features*2, 512, bias=True),
                            nn.ReLU(),
                            nn.BatchNorm1d(512, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True),
                            nn.Dropout(0.5),
                            nn.Linear(512, 1, bias=True))
            model._fc = new_fc
            print('gold fc added')

    if config.PARALLEL:
        model = nn.DataParallel(model)

    return model

