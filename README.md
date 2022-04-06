# Work 2: Convolutional neural networks (CNNs)

Universidade Federal do Paraná (UFPR)

**Team:** 
1. Diego Garzaro
2. Éder Hamasaki
3. Leonardo Bein
4. Natalia Choma
5. Vinícius Parede

## Part 01: Theoretical

### 1) What is deep learning? 
  Deep learning can be defined as a subcategory of machine learning where it's possible to learn by itself by many patterns in multiples processing layers, configured by basic parameters about the data base, based on the conding, experimenting and training data. 
  This means it's a learning algorithm with multiple levels of features that can leads to multiple levels of abstrations, similar to the as neural networks from the human brain, recognizing images, sounds, processing the human language accomplishing complex taks with no human intervention. 
  The advanteges consists on their robustness, scalatability and generalizable application.
  
  **References:**
  
  - [Deep Learning: O que é e qual sua importância?](https://www.sas.com/pt_br/insights/analytics/deep-learning.html)
  - [Deep Learning for Artificial Inteligence](https://pt.slideshare.net/ErShivaKShrestha/deep-learning-for-artificial-intelligence-ai)
  - [Afinal, o que é Deep Learning?](https://gaea.com.br/afinal-o-que-e-deep-learning/)

### 2) Why do we prefer CNNs over shallow artificial neural networks for image data?
  The CNN (Convolution Neural Networks) it's better because the convolutional layers can take advantage of inhererent properties of images, based on two steps: convolutions and pooling layers.
  Convolutions 
  
Convolutions
Simple feedforward neural networks don’t see any order in their inputs. If you shuffled all your images in the same way, the neural network would have the very same performance it had when trained on not shuffled images.
CNN, in opposition, take advantage of local spatial coherence of images. This means that they are able to reduce dramatically the number of operation needed to process an image by using convolution on patches of adjacent pixels, because adjacent pixels together are meaningful. We also call that local connectivity. Each map is then filled with the result of the convolution of a small patch of pixels, slid with a window over the whole image.

Pooling layers
There are also the pooling layers, which downscale the image. This is possible because we retain throughout the network, features that are organized spatially like an image, and thus downscaling them makes sense as reducing the size of the image. On classic inputs you cannot downscale a vector, as there is no coherence between an input and the one next to it.

Update 2021: In modern CNN architectures, pooling layers are replaced by strided convolutions.
  

### 3) Explain the role of the convolution layer in a CNN design.

Text Text Text

### 4) What is the role of the fully connected (FC) layer in CNN?


### 5) Why do we use a pooling layer in a CNN? 

Text Text Text

### 6) Explain the characteristics of the following pooling approaches: max pooling, average pooling, and sum pooling

Text Text Text
