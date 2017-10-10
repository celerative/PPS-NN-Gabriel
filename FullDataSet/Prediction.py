import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import csv
from keras.preprocessing import image

def get_labels (path):
    descriptions = []
    label_data = open(path, 'r') # annotations file
    label_reader = csv.reader(label_data) # csv parser for annotations file
    next(label_reader) # skip header
    for row in label_reader:
        descriptions.append((row[0], row[1]))
    return descriptions

def preprocess_image (path_img, net_name = ''):
    target_size = (64, 64)
    if (net_name == 'AlexNet'): target_size = (227, 227)
    if (net_name == 'ResNet'): target_size = (224, 224)
    if (net_name == 'Inception'): target_size = (299, 299)
    if (net_name == 'Xception'): target_size = (299, 299)
    if (net_name == 'Mobile'): target_size = (224, 224)
    img = image.load_img(path_img, target_size=target_size)
    img = image.img_to_array(img)
    img /= 255
    img = np.expand_dims(img, axis=0)
    return img

"""
    The path must contain 43 numbered folders
    verbose: print de predicted label with the % and the expected label
    create a file with the predictions missed
    
"""
def predict_from_directory (path_prefix, file_output, model, model_name='', verbose = 0):
    descriptions = get_labels ('./examples/predictions.csv')
    err_imgs = []
    with open(file_output, 'w', newline='') as f:
        writer = csv.writer(f)
        for label in range(43):
            path = path_prefix + str(label).zfill(2) + '/'
            imgs_names = [arch.name for arch in Path(path).iterdir() if arch.name.split(".")[-1] == 'jpg']
            for name in imgs_names:
                path_img = path + name
                img = preprocess_image (path_img, model_name)
                preds = model.predict(img)
                pos = preds.argmax()
                if (pos != label):
                    writer.writerow((path_img, pos, label))
                    if (verbose != 0):
                        print ('se predijo {} con {} acc en {} pero se esperaba {}'.format(
                                descriptions[pos], preds[0][pos], path_img, descriptions[label]))

def predict (path_img, model, l_expected='', model_name=''):
    img = preprocess_image (path_img, model_name)
    preds = model.predict(img)    
    index = np.argsort(preds[0])[::-1]
    preds = np.sort(preds[0])[::-1]
    i=0
    while (preds[i]>0.01):
        i+=1
    if (l_expected):
        f,bx = plt.subplots(1, 2)
        bx[0].imshow(Image.open(path_img))
        bx[0].axes.get_xaxis().set_visible(False)
        bx[0].axes.get_yaxis().set_visible(False)
        bx[1].imshow(Image.open('./examples/images/' + str(l_expected) + '.jpg'))
        bx[1].axes.get_xaxis().set_visible(False)
        bx[1].axes.get_yaxis().set_visible(False)
    else:
        plt.imshow(Image.open(path_img))
        
    col = i//4 if (i%4 == 0) else i//4 + 1
    if (col > 1):
        f,ax = plt.subplots(col, 4)
        for j in range(col*4):
            ax[j//4][j%4].axes.get_xaxis().set_visible(False)
            ax[j//4][j%4].axes.get_yaxis().set_visible(False)
        for j in range(i):
            ax[j//4][j%4].imshow(Image.open('./examples/images/' + str(index[j]) + '.jpg'))
    else:
        if (i > 1):
            f,ax = plt.subplots(1, i)
            for j in range(i):
                ax[j].axes.get_xaxis().set_visible(False)
                ax[j].axes.get_yaxis().set_visible(False)
                ax[j].imshow(Image.open('./examples/images/' + str(index[j]) + '.jpg'))
        else:
            plt.show()
            plt.imshow(Image.open('./examples/images/' + str(index[0]) + '.jpg'))
    plt.show()
    for j in range(i):
        print ('image {} predicted with {}%'.format(j, preds[j]*100))
    return np.array([index, preds]).T

def show_predicted (path_img, l_predicted, l_expected):
    #print ('Format: Original - - Expected - - Predicted')
    ori_img = Image.open(path_img)
    exp_img = Image.open('./examples/images/' + str(l_expected) + '.jpg')
    pre_img = Image.open('./examples/images/' + str(l_predicted) + '.jpg')
    f,ax = plt.subplots(1,3)
    ax[0].imshow(ori_img)
    ax[1].imshow(exp_img)
    ax[2].imshow(pre_img)
    plt.show()
    """
    plt.imshow(ori_img)
    plt.imshow(pre_img)
    plt.imshow(exp_img)
    plt.show()
    """
def show_predicted_csv (path_csv):
    print ('Format: Original - - Predicted - - Expected')
    with open(path_csv, 'r') as archivo:
        labels = csv.reader(archivo, delimiter=',')
        for row in labels:
            show_predicted (row[0], row[1], row[2])
