import cv2
import time
import uuid
import os
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

#banco de imagens

WORKSPACE_PATH = 'librasVC/workspace'
SCRIPTS_PATH = 'librasVC/scripts'
APIMODEL_PATH = 'librasVC/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

#dicio das palavras
labels = ['Oi', 'Obrigado', 'Sim', 'Não', 'Eu_te_amo']
num_imgs = 15

## labels map
labels = [
    {'name': 'oi', 'id': 1},
    {'name': 'não', 'id': 2},
    {'name': 'sim', 'id': 3},
    {'name': 'eu te amo', 'id': 4},
    {'name': 'obrigado', 'id': 5},
]

with open(ANNOTATION_PATH + '/label_map.txt', 'w') as f:
    for label in labels:
        f.write('item {\n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


#{SCRIPTS_PATH + '/generate_tfrecord.py'} - x{IMAGE_PATH + '/train'} - l{ANNOTATION_PATH + '/label_map.pbtxt'} - o{ANNOTATION_PATH + '/train.record'}
#{SCRIPTS_PATH + '/generate_tfrecord.py'} - x{IMAGE_PATH + '/test'} - l{ANNOTATION_PATH + '/label_map.pbtxt'} - o{ANNOTATION_PATH + '/test.record'}

for label in labels:
    caminho = ('images\collectedimages\\{}'.format(label))
    os.mkdir(caminho)
    #instanciando a camera
    cap = cv2.VideoCapture(0)
    print('coleções de imagens para {}'.format(label))
    time.sleep(5)
    for imgnum in range(num_imgs):
        ret, frame = cap.read()
        image_name = os.path.join(IMAGE_PATH , label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        print(image_name)
        #salvando foto
        cv2.imwrite(image_name, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

