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
  This means it's a learning algorithm with multiple levels of features that can leads into multiple levels of abstrations, similar to the as neural networks from the human brain, recognizing images, sounds, processing the human language accomplishing complex taks with no human intervention. The advanteges consists on their robustness, scalatability and generalizable application.
  
  **References:**
  
  - [Deep Learning: what is and importance?](https://www.sas.com/pt_br/insights/analytics/deep-learning.html)
  - [Deep Learning for Artificial Inteligence](https://pt.slideshare.net/ErShivaKShrestha/deep-learning-for-artificial-intelligence-ai)
  - [What's Deep Learning?](https://gaea.com.br/afinal-o-que-e-deep-learning/)

### 2) Why do we prefer CNNs over shallow artificial neural networks for image data?
   The definition of convolutional neural network (CNN) is a type of artificial neural network usually used in image recognition and processing, that use deep learning to perform both generative and descriptive tasks. That's the reason why it's better, the convolutional layers can take an advantage of inhererent properties of images, based on two steps: convolutions and pooling layers.
  Convolutions it's the step where feedfoward network it doesn't differentiate the input order, shuffling the all images, the neural network can have the smae performance compared to a not shuffled image. Basically, it's possible to reduce the number of operation required leading to the second step: pooling layers.
  By the poooling layers it's possible to downscale the image, retaining throughout the network, maintaining the coherence between an input and the one next to it, when it's compared to on classic inputs.

**References**
-[Why are convolutional neural networks better than other neural networks in processing data such as images and video?](https://www.quora.com/Why-are-convolutional-neural-networks-better-than-other-neural-networks-in-processing-data-such-as-images-and-video)
-[Convolutional Neural Network](https://www.techtarget.com/searchenterpriseai/definition/convolutional-neural-network)  

### 3) Explain the role of the convolution layer in a CNN design.
The main role for convolutional layer it's to build block of convolution neural networks, where contains filters or parameters to be learned. When the filter application are repeated for an input results in the featured map, it can be possible to see the locations and streghth for a detected feature of an input. It's possible to obtain a vector representation of some object of the found net, summarizing, the convolution layers can build features from raw data. 

**Refrences**
-[Training Convolutional Nets to Detect Calcified Plaque in IVUS Sequences](https://www.sciencedirect.com/topics/engineering/convolutional-layer#:~:text=A%20convolutional%20layer%20is%20the,and%20creates%20an%20activation%20map.)
-[How Do Convolutional Layers Work in Deep Learning Neural Networks?](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)

### 4) What is the role of the fully connected (FC) layer in CNN?
Fully connected layers is defined by the last few layers in the network or the feed foward neural networks. It's the output from the final pooling or convolutional layer, in which is flatter and fed into the connected layer. It's resulted an flattened vector connected to a fully conected layers in which can perform mathematical operation the same as artificial neural networks. The final layers is used for the activation function where it get probabilities of the input used in a speficic class.

**References**
-[Convolutional Neural Network](https://towardsdatascience.com/convolutional-neural-network-17fb77e76c05#:~:text=Fully%20Connected%20Layer,-Fig%204.&text=Fully%20Connected%20Layer%20is%20simply,into%20the%20fully%20connected%20layer.)

### 5) Why do we use a pooling layer in a CNN? 
Pooling layers is used because it's possible to reduce the dimension of the featured maps, reducing the numbers of parameters to learn and computation performance in the network.
Another reason it's because the feature map can be generated by convolution layer. It's a robust model where the position of the features in the input ins't necessary to have positioned features genereated.

**References**
-[CNN | Introduction to Pooling Layer](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/#:~:text=Why%20to%20use%20Pooling%20Layers,generated%20by%20a%20convolution%20layer.)

### 6) Explain the characteristics of the following pooling approaches: max pooling, average pooling, and sum pooling
Knowing that pooling layers for convolutional neural network is a learned filter to input images, it can provides an approach to down feature maps by summarizing the presence of features inpatches of the featured map. When it commes to create feature maps there are two methods that summarize the average presence of a feature:
  Max pooling: it's a fucntion that calculates the maximum value for each patch of the feature map, it shows the most activated presence of a feature.
  Average poooling: it's a function that calculate the average value for the featured map, it shows the average presence of a feature.
  Sum pooling: it's a function that calculates the sum value for the featured map


**References**
-[A Gentle Introduction to Pooling Layers for Convolutional Neural Networks](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)
