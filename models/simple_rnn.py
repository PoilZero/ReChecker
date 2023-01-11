from __future__ import print_function

import warnings
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, ReLU, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adamax
from sklearn.model_selection import train_test_split
from P001_parser import parameter_parser

args = parameter_parser()

warnings.filterwarnings("ignore")

"""
Baseline FC
"""


class Simple_RNN:
    def __init__(self, data, name="", batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, dropout=args.dropout,
                 threshold=args.threshold):
        vectors = np.stack(data.iloc[:, 0].values)
        labels = data.iloc[:, 1].values
        x_train, x_test, y_train, y_test = train_test_split(vectors, labels,
                                                            test_size=0.2, stratify=labels)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = threshold
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)

        model = Sequential([
            SimpleRNN(300, input_shape=(vectors.shape[1], vectors.shape[2])) # dim

            , Dropout(dropout)
            , Dense(2, activation='softmax')
        ])

        # Lower learning rate to prevent divergence
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # set lr
        adamax = Adamax(lr)
        model.compile(optimizer=adamax, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Trains model
    """

    def train(self):
        self.model.fit(
            self.x_train, self.y_train
            , batch_size=self.batch_size
            , epochs=self.epochs
            , class_weight=self.class_weight
            , verbose=2
        )

    """
    Tests accuracy of model
    Loads weights from file if no weights are attached to model object
    """

    def test(self):
        values = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size, verbose=1)
        print("Accuracy: ", values[1])
        predictions = (self.model.predict(self.x_test, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall: ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
        print()
