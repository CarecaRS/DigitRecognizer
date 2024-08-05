# Kaggle Challenge
# Digit Recognizer

This challenge aims to indentify handwritten digits images using computer vision. For this, the Kaggle chalenge partially uses the MNIST digits classification dataset. The whole database is available within TensorFlow/Keras packages for Python.

## My solution
My model is made using TensorFlow, based on a sequential model for analyzing information from the database. I use a small convolutional neural network (CNN) to solve the challenge, with 9 layers in total. This total already includes the input and output layers, the hidden layers are composed of convolutional layers, max pooling layers for 2D data, a flattening layer, a dense processing layer and the dense layer for output, in addition to a dropout layer for regularization and prevention/adjustment of overfitting. The code is developed entirely in Python using the `neovim` application.

In addition to the code for the actual solution to the problem, my code records the models to enable later loading if necessary, also creating/updating a report file with the information I deemed necessary for retroactive analysis of each of the models tested, with their relevant parameters and scores ('accuracy' analysis criteria).

## Final score and classification
Through my model, I achieved a score of 0.99757 (with a maximum of 1.0 representing 100% accuracy) on 05/aug/2024, ranking me in position 42 out of 1,798 participants on that date, which placed me in the top 3% of the best models in the challenge.
