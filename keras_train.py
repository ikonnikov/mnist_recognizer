# -*- coding: utf-8 -*-

from keras_recognizer import Recognizer
from keras.datasets import mnist
import matplotlib.pylab as plt

img_x = 28
img_y = 28
batch_size = 100
epochs = 10


def sized_recognizer(train_data, test_data, model_filename, weights_filename):
    recognizer = Recognizer(batch_size=batch_size, epochs=epochs, img_x=img_x, img_y=img_y)
    recognizer.load_data(train_data=train_data, test_data=test_data)
    recognizer.build_model()
    recognizer.train_model(verbose=0)

    accuracy = recognizer.evaluate_model()

    recognizer.save_model(model_filename=model_filename, weights_filename=weights_filename)

    return accuracy


def main():
    print('Download MNIST-data...')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print('Start recognize: 100% size pics')

    accuracy_100 = sized_recognizer((x_train, y_train), (x_test, y_test), 'size100_model.json', 'size100_model_weights.h5')

    print('---> accuracy for 100% size pics: {}'.format(accuracy_100))
    print('\n')

    print('Start recognize: 75% size pics')

    x_train[:, int(img_y * 0.75):] = 0
    x_test[:, int(img_y * 0.75):] = 0

    accuracy_75 = sized_recognizer((x_train, y_train), (x_test, y_test), 'size75_model.json', 'size75_model_weights.h5')

    print('---> accuracy for 75% size pics: {}'.format(accuracy_75))
    print('\n')

    print('Start recognize: 50% size pics')

    x_train[:, int(img_y * 0.50):] = 0
    x_test[:, int(img_y * 0.50):] = 0

    accuracy_50 = sized_recognizer((x_train, y_train), (x_test, y_test), 'size50_model.json', 'size50_model_weights.h5')

    print('---> accuracy for 50% size pics: {}'.format(accuracy_50))
    print('\n')

    print('Start recognize: 25% size pics')

    x_train[:, int(img_y * 0.25):] = 0
    x_test[:, int(img_y * 0.25):] = 0

    accuracy_25 = sized_recognizer((x_train, y_train), (x_test, y_test), 'size25_model.json', 'size25_model_weights.h5')

    print('---> accuracy for 25% size pics: {}'.format(accuracy_25))

    accuracies = [accuracy_100, accuracy_75, accuracy_50, accuracy_25]

    plt.plot([100, 75, 50, 25], accuracies)
    plt.xlabel('Pic size, %')
    plt.ylabel('Accuracy')
    plt.grid(True)
    # plt.show()
    plt.savefig(filename='sized_accuracy_fig.png', format='png')


if __name__ == '__main__':
    main()
