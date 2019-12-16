# styletransfernetwork

Style Transfer Network implemented based on https://arxiv.org/abs/1603.08155 and https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf


pytorch basics to understand the code: https://pytorch.org/tutorials/


pre-trained models are located in the models folder


This style transfer network attempts to train a generator to produce stylized images in the style of a single input image. The generator is trained to minimize the difference between the vgg 16 feature maps of its output and of the style image.

FILES:
generator.py: file containing class definition for the generator implemented as described in the supplemental material

superres.py: file containing class definition of a network that is supposed to create higher resolution images from low resolution images

train.py: training loop for the generator network

train_super_res.py: training loop for super resolution network

not mine:

  vgg.py: taken from the official codebase, class definition for the "discriminator"

FOLDERS:
style images folder: contains images of some of the styles I used to train the generator

results folder: contains successfully styled images

