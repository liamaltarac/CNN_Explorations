import numpy as np
from scipy import ndimage


from tensorflow.nn import depthwise_conv2d
from tensorflow.math import multiply, reduce_sum, reduce_mean,reduce_euclidean_norm, sin, cos, abs
from tensorflow import stack, concat, expand_dims

import tensorflow_probability as tfp
from scipy.fftpack import dct, idct


def get_filter(model, layer):

    conv_layers = []
    for l in model.layers:
        if 'conv2d' in str(type(l)).lower():
            if l.kernel_size == (3,3):
                conv_layers.append(l)

    layer = conv_layers[layer]
    # check for convolutional layer
    if 'conv' not in layer.name:
        raise ValueError('Layer must be a conv. layer')
    # get filter weights
    filters, biases = layer.get_weights()
    #print("biases shape : ", biases.shape)
    #print("filters shape : ", filters.shape)
    #print(layer.name)
    return (filters)
    #print(layer.name, filters.shape)



def getSobelTF(f):

    sobel_h = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=np.float32).reshape((3, 3, 1, 1) )/-4
    sobel_v = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], dtype=np.float32).reshape((3, 3, 1, 1))/-4    

    s_h = reduce_sum(multiply(f, sobel_h), axis=[0,1])
    s_v = reduce_sum(multiply(f, sobel_v), axis=[0,1])

    return (np.arctan2(s_h,s_v))


def getSymAntiSymTF(filter):

    #patches = extract_image_patches(filters, [1, k, k, 1],  [1, k, k, 1], rates = [1,1,1,1] , padding = 'VALID')
    #print(patches)
    a = filter[0,0,:,:]
    b = filter[0,1,:,:]
    c = filter[0,2,:,:]
    d = filter[1,0,:,:]
    e = filter[1,1,:,:]
    f = filter[1,2,:,:]
    g = filter[2,0,:,:]
    h = filter[2,1,:,:]
    i = filter[2,2,:,:]

    fs1 = expand_dims(a+c+g+i, 0)/4
    fs2 = expand_dims(b+d+f+h,0)/4
    fs3= expand_dims(e, 0)

    sym = stack([concat([fs1, fs2, fs1],  axis=0), 
                         concat([fs2, fs3, fs2], axis=0),
                         concat([fs1, fs2, fs1], axis=0)])
        
    anti = filter - sym

    return sym, anti

def topKfilters(model, layer_num, k=10):
    #print(i, l.name)
    filters = get_filter(model, layer_num)

    mag = reduce_euclidean_norm(filters, axis=[0,1])**2
    avg_mag = reduce_mean(mag, axis=0)
    idx = list(range(mag.shape[-1]))
    
    idx = [x for _, x in sorted(zip( avg_mag, idx), reverse=True)]
    return idx[:int(np.floor(len(idx)*k/100))]

def topKchannels(model, layer_num, f_num, k=10):
    #print(i, l.name)
    filters = get_filter(model, layer_num)[:,:,:,f_num]

    mag = reduce_euclidean_norm(filters, axis=[0,1])**2
    #avg_mag = reduce_mean(mag, axis=0)
    idx = list(range(mag.shape[-1]))
    if int((k/100)*len(idx)) == 0:
        return idx
    
    idx = [x for _, x in sorted(zip( mag, idx), reverse=True)]
    return idx[:int(np.floor(len(idx)*k/100))]


def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    