from VGG16_test_leo import run_exp as VGG
from Inceptionv3_test_leo import run_exp as InceptionV3
from Inceptionv4_test_leo import run_exp as InceptionV4
from ResNet50_test_leo import run_exp as ResNet50
from DenseNet_test_leo import run_exp as DenseNet

VGG([['./log/VGG fixed/', 3, 1, 16], ['./log/VGG fixed x1/', 3, 1, 2], ['./log/VGG fixed x2/', 3, 2, 2], ['./log/VGG fixed x3/', 3, 3, 2]])
InceptionV3([['./log/Inception V3 fixed/', 3, 1, 16], ['./log/Inception V3 fixed x1/', 3, 1, 2], ['./log/Inception V3 fixed x2/', 3, 2, 2], ['./log/Inception V3 fixed x3/', 3, 3, 2]])
InceptionV4([['./log/Inception V4 fixed/', 3, 1, 16], ['./log/Inception V4 fixed x1/', 3, 1, 2], ['./log/Inception V4 fixed x2/', 3, 2, 2], ['./log/Inception V4 fixed x3/', 3, 3, 2]])
ResNet50([['./log/ResNet fixed/', 3, 1, 16], ['./log/ResNet fixed x1/', 3, 1, 2], ['./log/ResNet fixed x2/', 3, 2, 2], ['./log/ResNet fixed x3/', 3, 3, 2]])
DenseNet([['./log/DenseNet fixed/', 3, 1, 16], ['./log/DenseNet fixed x1/', 3, 1, 2], ['./log/DenseNet fixed x2/', 3, 2, 2], ['./log/DenseNet fixed x3/', 3, 3, 2]])
