Image Captioning with Attention using PyTorch

This repository contains code for an image captioning model with attention mechanism implemented in PyTorch. The model generates descriptive captions for images by attending to different parts of the image while generating each word of the caption.

Overview

The image captioning model consists of two main components:

Encoder: A ResNet CNN pre-trained on ImageNet is used to extract features from input images. These features are then passed to the decoder.

Decoder: An LSTM-based decoder generates captions word by word. It utilizes an attention mechanism to focus on different parts of the image at each step of caption generation.

Requirements

Python 3.x
PyTorch
NLTK
NumPy
Matplotlib
torchvision
tqdm
Pillow
scikit-image
pycocotools
imageio
pytorch_pretrained_bert
You can install the required packages using pip:
pip install -r requirements.txt

Usage

Data Preparation:

Download the COCO 2017 dataset and annotations from here.
Organize the dataset and annotations in the required directory structure. You can adjust the paths accordingly in the code.
Building Vocabulary:

Run the main function in the provided script to build the vocabulary.

caption_path = 'path_to_annotations_file'
vocab_path = 'path_to_save_vocab.pkl'
threshold = 5
main(caption_path, vocab_path, threshold)


Training:

Use the get_loader function to create data loaders for training and validation sets.
Initialize the encoder and decoder models.
Define the loss function and optimizer.
Train the model using the provided training loop.

Evaluation:

Evaluate the model using metrics like BLEU score on a separate validation set.
Generate captions for test images and visualize the results.
