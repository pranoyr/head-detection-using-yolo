import pickle
import numpy as np
import cv2
import os


def dataset_train():
    X = []
    y = []
    for selected_label in labels:
        all_images = os.listdir(dir_train_dataset + selected_label)
        for image in all_images:
            img = cv2.imread(dir_train_dataset + selected_label + "/" + image)
            # img = cv2.resize(img, (128, 128))
            X.append(img)
            y.append(labels.index(selected_label))

    X = np.array(X)
    y = np.array(y)
    return X, y


def dataset_dev():
    X = []
    y = []
    for selected_label in labels:
        all_images = os.listdir(dir_dev_dataset + selected_label)
        for image in all_images:
            img = cv2.imread(dir_dev_dataset + selected_label + "/" + image)
            # img = cv2.resize(img, (128, 128))
            X.append(img)
            y.append(labels.index(selected_label))

    X = np.array(X)
    y = np.array(y)
    return X, y


# dataset directory
dir_train_dataset = "dataset/tr`ain set/"
dir_dev_dataset = "dataset/dev set/"

labels = os.listdir(dir_train_dataset)
pickle.dump(labels, open("int2word_out.pkl", "wb"))
