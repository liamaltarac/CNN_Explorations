import sys
sys.path.append('../')

import gc
import numpy as np
from scipy import ndimage

from skimage.filters import sobel_h
from skimage.filters import sobel_v
from scipy import stats

#from sa_decomp_layer import SADecompLayer



'''
Strange results for gradient tape : Getting positive gradients for negative response

Hello,

I'm working with some gradient based interpretability method ([based on the GradCam code from Keras ](https://keras.io/examples/vision/grad_cam/) ) , and I'm running into a result that seems inconsistent with what would expect from backpropagation.  

I am working with a pertrained VGG16 on imagenet, and I am interested in find the most relevent filters for a given class.

I start by forward propagating an image through the network, and then from the relevant bin, I find the gradients to the layer in question (just like they do in the Keras tutorial).

Then, from the pooled gradients (`pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))`), I find the top-K highest/most pertinent filters. 

From this experiment, I run into 2 strange results. 

1.  For almost any image I pass through (even completely different classes), the network almost always seems to be placing the most importance to the same 1 Filter.

2.  And this result I understand even less; many times, the gradients point "strongly" to a filter, even though the filter's output is 0/negative (before relu).  From the backpropagation equation, a negative response should result in a Null gradient, right ?

$$ \frac{dY_{class}}{dActivation} = accumulated\\_gradients \cdot \frac{dY}{dActivation}$$
 $$ = accumulated\\_gradients \cdot Relu'(I*w+b)$$

If $I*w+b$ is negative, then $\frac{dY_{class}}{dActivation}$ should be 0, right ? 




'''
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications import VGG16
import keras
from keras import backend as K

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

img = "img3.JPEG"
img = keras.applications.vgg16.preprocess_input(get_img_array(img, size=(224,224)))

model = VGG16(weights='imagenet',
				  include_top=True,
				  input_shape=(224, 224, 3))

# Remove last layer's softmax
model.layers[-1].activation = None


#I am interested in finding the most informative filters from this Layer
layer = model.get_layer("block5_conv3")

grad_model = keras.models.Model(
    model.inputs,  [layer.output, model.output]
)


pred_idx = None
with tf.GradientTape(persistent=True) as tape:
    last_conv_layer_output, preds = grad_model(img, training=False)

    if pred_idx is None:
        pred_idx = tf.argmax(preds[0])
    print(tf.argmax(preds[0]))
    print(decode_predictions(preds.numpy()))
    class_channel = preds[:, pred_idx]


grads = tape.gradient(class_channel,  last_conv_layer_output) #
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
topFilters = tf.math.top_k(pooled_grads, k=5).indices.numpy()

print("Top Filters : ", topFilters)
print("Filter responses: " , tf.math.reduce_euclidean_norm(last_conv_layer_output, axis=(0,1,2)).numpy()[topFilters])

plt.imshow(last_conv_layer_output[0,:,:,336])
plt.show()



