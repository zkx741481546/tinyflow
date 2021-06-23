from VGG16_test_leo import run_exp as VGG
from Inceptionv3_test_leo import run_exp as InceptionV3
from Inceptionv4_test_leo import run_exp as InceptionV4
from ResNet50_test_leo import run_exp as ResNet50
from DenseNet_test_leo import run_exp as DenseNet

VGG([['./log/VGG bs1/', 3, 1, 1], ['./log/VGG bs2/', 3, 1, 2], ['./log/VGG bs4/', 3, 1, 4], ['./log/VGG bs8/', 3, 1, 8], ['./log/VGG bs32/', 3, 1, 32]])
InceptionV3([['./log/InceptionV3 bs1/', 3, 1, 1], ['./log/InceptionV3 bs2/', 3, 1, 2], ['./log/InceptionV3 bs4/', 3, 1, 4], ['./log/InceptionV3 bs8/', 3, 1, 8], ['./log/InceptionV3 bs32/', 3, 1, 32]])
InceptionV4([['./log/InceptionV4 bs1/', 3, 1, 1], ['./log/InceptionV4 bs2/', 3, 1, 2], ['./log/InceptionV4 bs4/', 3, 1, 4], ['./log/InceptionV4 bs8/', 3, 1, 8], ['./log/InceptionV4 bs32/', 3, 1, 32]])
ResNet50([['./log/ResNet50 bs1/', 3, 1, 1], ['./log/ResNet50 bs2/', 3, 1, 2], ['./log/ResNet50 bs4/', 3, 1, 4], ['./log/ResNet50 bs8/', 3, 1, 8], ['./log/ResNet50 bs32/', 3, 1, 32]])
DenseNet([['./log/DenseNet bs1/', 3, 1, 1], ['./log/DenseNet bs2/', 3, 1, 2], ['./log/DenseNet bs4/', 3, 1, 4], ['./log/DenseNet bs8/', 3, 1, 8], ['./log/DenseNet bs32/', 3, 1, 32]])