# -*- coding: utf-8 -*-
from __future__ import print_function

import time

import numpy as np
from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b


#########################################################
class P(object):

    def __init__(self, base_image_path, style_reference_image_path, result_prefix, iter=10, tv_weight=1.0,
                 style_weight=1.0,
                 content_weight=0.025):
        self.base_image_path = base_image_path
        self.style_reference_image_path = style_reference_image_path
        self.result_prefix = result_prefix
        self.iterations = iter

        # these are the weights of the different loss components
        self.total_variation_weight = tv_weight
        self.style_weight = style_weight
        self.content_weight = content_weight

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(self.img_nrows, self.img_ncols))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg16.preprocess_input(img)
        return img

    # util function to convert a tensor into a valid image

    def deprocess_image(self, x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, self.img_nrows, self.img_ncols))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((self.img_nrows, self.img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def gram_matrix(self, x):
        assert K.ndim(x) == 3
        if K.image_data_format() == 'channels_first':
            features = K.batch_flatten(x)
        else:
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    def style_loss(self, style, combination):
        assert K.ndim(style) == 3
        assert K.ndim(combination) == 3
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        size = self.img_nrows * self.img_ncols
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def content_loss(self, base, combination):
        return K.sum(K.square(combination - base))

    def total_variation_loss(self, x):
        assert K.ndim(x) == 4
        if K.image_data_format() == 'channels_first':
            a = K.square(x[:, :, :self.img_nrows - 1, :self.img_ncols - 1] - x[:, :, 1:, :self.img_ncols - 1])
            b = K.square(x[:, :, :self.img_nrows - 1, :self.img_ncols - 1] - x[:, :, :self.img_nrows - 1, 1:])
        else:
            a = K.square(x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] - x[:, 1:, :self.img_ncols - 1, :])
            b = K.square(x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] - x[:, :self.img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    def eval_loss_and_grads(self, x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, self.img_nrows, self.img_ncols))
        else:
            x = x.reshape((1, self.img_nrows, self.img_ncols, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values


class Evaluator(object):

    def __init__(self, picture_class):
        self.loss_value = None
        self.grads_values = None
        self.pic = picture_class

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.pic.eval_loss_and_grads(x)
        # loss_value, grad_values = P.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


##########################################################################################


def p_main(base_image_path1, style_reference_image_path1, result_prefix1):
    base_image_path = base_image_path1
    style_reference_image_path = style_reference_image_path1
    result_prefix = result_prefix1
    # iterations = 10

    p = P(base_image_path, style_reference_image_path, result_prefix)

    # these are the weights of the different loss components
    # total_variation_weight = 1.0
    # style_weight = 1.0
    # content_weight = 0.025

    # dimensions of the generated picture.
    width, height = load_img(base_image_path).size
    p.img_nrows = 400
    p.img_ncols = int(width * p.img_nrows / height)

    # util function to open, resize and format pictures into appropriate tensors

    # get tensor representations of our images
    base_image = K.variable(p.preprocess_image(base_image_path))
    style_reference_image = K.variable(p.preprocess_image(style_reference_image_path))

    # this will contain our generated image
    if K.image_data_format() == 'channels_first':
        combination_image = K.placeholder((1, 3, p.img_nrows, p.img_ncols))
    else:
        combination_image = K.placeholder((1, p.img_nrows, p.img_ncols, 3))

    # combine the 3 images into a single Keras tensor
    input_tensor = K.concatenate([base_image,
                                  style_reference_image,
                                  combination_image], axis=0)

    # build the VGG16 network with our 3 images as input
    # the model will be loaded with pre-trained ImageNet weights
    model = vgg16.VGG16(input_tensor=input_tensor,
                        weights='imagenet', include_top=False)
    print('Model loaded.')

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # compute the neural style loss
    # first we need to define 4 util functions

    # the gram matrix of an image tensor (feature-wise outer product)
    ####################################################################################
    # combine these loss functions into a single scalar
    loss = K.variable(0.)
    layer_features = outputs_dict['block4_conv2']
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += p.content_weight * p.content_loss(base_image_features,
                                              combination_features)

    feature_layers = ['block1_conv1', 'block2_conv1',
                      'block3_conv1', 'block4_conv1',
                      'block5_conv1']
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = p.style_loss(style_reference_features, combination_features)
        loss += (p.style_weight / len(feature_layers)) * sl
    loss += p.total_variation_weight * p.total_variation_loss(combination_image)

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)

    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    p.f_outputs = K.function([combination_image], outputs)

    #############################################################################################

    evaluator = Evaluator(p)

    # run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss
    if K.image_data_format() == 'channels_first':
        x = np.random.uniform(0, 255, (1, 3, p.img_nrows, p.img_ncols)) - 128.
    else:
        x = np.random.uniform(0, 255, (1, p.img_nrows, p.img_ncols, 3)) - 128.

    for i in range(p.iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # save current generated image
        img = p.deprocess_image(x.copy())
        fname = p.result_prefix + '_at_iteration_%d.png' % i
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
    img_save = imsave(fname, img)
    print('img_save', img_save, dir(img_save))
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    return fname


# if __name__ == '__main__':
    # p_main('', '', '')
