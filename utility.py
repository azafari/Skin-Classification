def train_test_shuffled_separation(data, label, train_percent=0.8):
    """
    """
    import numpy as np

    # Randomize training set and corresponding labels
    rand_set = np.hstack((label, data))
    np.random.shuffle(rand_set)
    data = rand_set[:, 1:]
    label = rand_set[:, 0]
    print("shuffled data shape:", data.shape, "shuffled label shape:", label.shape)

    # specify train and test sizes
    train_length = int(train_percent * len(data))

    # index first 80% for training, last 20% for test
    data_train = data[0: train_length, :]
    label_train = label[0: train_length]

    data_test = data[train_length:, :]
    label_test = label[train_length:]

    print('# train:', len(data_train))
    print('# test: ', len(data_test))
    print('# total:', len(data))

    return data_train, label_train, data_test, label_test


def normalize_data(data, min=0, max=1):
    """
    takes a collection of images (such as xtrain) and return a modified version in the range [-1, 1]
    of type float64.
    :param x (np.ndarray): collection of images in 2d array, each image index is x[:, n]
    :return: (np.ndarray) normalized collection of images in type float64
    """
    import numpy as np
    assert isinstance(data, np.ndarray)

    max_value = np.max(data)
    min_value = np.min(data)

    scaled = np.interp(data, [min_value, max_value], [min, max])
    # convert to float64
    scaled = scaled.astype(np.float64)

    return scaled


def label_to_one_hot(label):
    """
    converts 1d array to 2d binary one hot labels
    (n, ) to (n, 2)
    :param label:
    :return:
    """
    import numpy as np
    num_of_class = int(np.max(label))
    one_hot = np.zeros((len(label), num_of_class), dtype=np.uint8)
    for i in range(len(label)):
        one_hot[i, int(label[i] - 1)] = 1  # label is 1 and 2

    return one_hot


def plot_history(history, name=""):
    """

    :param history:
    :return:
    """
    import matplotlib.pyplot as plt
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['top_2_accuracy'])
    plt.plot(history.history['val_top_2_accuracy'])
    plt.title(name + 'Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation', 'Training top-2', 'Validation top-2'], loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name + 'Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.show()


def list_images(data, label, classes, cmap=None):
    """
    function to display the images
    :param data:
    :param label:
    :param ylabel:
    :param cmap:
    :return:
    """
    import matplotlib.pyplot as plt
    import random

    plt.figure(figsize=(15, 16))

    for i in range(6):
        plt.subplot(1, 6, i + 1)
        indx = random.randint(0, len(data))

        # Use gray scale color map if there is only one channel
        cmap = 'gray' if len(data[indx].shape) == 2 else cmap
        plt.imshow(data[indx], cmap=cmap)

        plt.xlabel(classes[label[indx]])
        # plt.ylabel(ylabel)

        plt.xticks([])
        plt.yticks([])
    # plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


if __name__ == "__main__":
    import numpy as np

    # labels = np.ones((10))*2
    # hot = label_to_one_hot(labels)
    pass
