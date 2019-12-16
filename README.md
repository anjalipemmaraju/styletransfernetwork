# style-transfer-network
### Background

Style Transfer Network and Super Resolution Network implemented based on https://arxiv.org/abs/1603.08155 and https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf

This implementation uses PyTorch, which is a machine learning framework. The official tutorials are linked: https://pytorch.org/tutorials/.

This code solves 2 problems: the first part of this code stylizes an image based on an input style. The second part of the code tries to generate a higher resolution image from a low resolution image.

The generator is a network that tries to produce "good" stylized images that maintain the original content of its input. The "goodness" of the image is defined by the discriminator and its feature maps. In this case, the discriminator is a pre-trained VGG16 network downloaded from the PyTorch website. The loss function is made up of two parts that represent the "style loss" and the "content loss", and is calculated as a difference between the discriminator's feature maps of the generated output image and of the style image. Based on this loss, the generator network's weights will get updated to try to minimize the difference in the feature maps.

For example, given this content-image of a dog, and this style-image of Van Gogh's starry night, the generator is trained to output an image that blends the content-image and the style-image.

<img src="https://github.com/anjalipemmaraju/styletransfernetwork/blob/master/content_images/original_dog.jpg" width="256"> <img src="https://github.com/anjalipemmaraju/styletransfernetwork/blob/master/style_images/vangogh.jpg" width="256">
<img src="https://github.com/anjalipemmaraju/styletransfernetwork/blob/master/results/vangogh_dog.jpg" width="512">

The super resolution network training loop uses paired low resolution and high resolution images as training input and again. The network takes in low resolution images and tries to produce outputs that when run through VGG16 has a similar feature map to the high resolution paired training image. This code is still in progress.

### Code
1. generator.py: file containing class definition for the generator implemented as described in the supplemental material
2. superres.py: file containing class definition of a network that is supposed to create higher resolution images from low resolution images
3. run.py: Defines a training loop for the generator network. This file also has a function to convert test images and test videos to stylized images and videos.
4. run_super_res.py: Defines a training loop for super resolution network and a test function to see if images are correctly brought to higher resolution.
5. vgg.py: taken from the official codebase, class definition for the "discriminator". https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py

### FOLDERS:
1. models folder: contains some pre-trained generator models
2. content_images folder: contains the image of the dog that was stylized
3. style_images folder: contains images of some of the styles I used to train the generator
4. results folder: contains successfully stylized images

