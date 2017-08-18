import os.path as osp
import sys
import copy
import os
import numpy as np
import google.protobuf as pb

import sys
sys.path.append('/home/yuepan/code/caffe/python')
import caffe
import caffe.proto.caffe_pb2 as cp
import time

caffe.set_mode_cpu()
layer_type = ['Convolution', 'InnerProduct']
bnn_type = ['BatchNorm', 'Scale']
temp_file = './temp.prototxt'
eps=1e-5

def get_netparameter(model):
  with open(model) as f:
    net = cp.NetParameter()
    pb.text_format.Parse(f.read(), net)
    return net


if __name__ == '__main__':
  
  model = './models/deploy_model_1.prototxt'
  weights = './models/model_1.caffemodel'
  dest_model_dir = './models/result_model_1.prototxt'
  dest_weight_dir = './models/result_model_1.caffemodel'
  
  '''
  model = './models/deploy_cifar10_bn.prototxt'
  weights = './models/cifar10_bn_iter_2000.caffemodel'
  dest_model_dir = './models/result_model_2.prototxt'
  dest_weight_dir = './models/result_model_2.caffemodel'
  '''
  '''
  model = './models/deploy_68_new.prototxt'
  weights = './models/vgg_68_new.caffemodel'
  dest_model_dir = './models/result_model_3.prototxt'
  dest_weight_dir = './models/result_model_3.caffemodel'
  '''
  net_model = caffe.Net(model, weights, caffe.TEST)
  net_param = get_netparameter(model)

  model_layers = net_model.layers
  '''
  print("model_layers",len(model_layers))
  for num in range(len(model_layers)):
    print model_layers[num].type 
  '''
  param_layers = net_param.layer
  '''
  print("param_layers:",len(param_layers))
  for num in range(len(param_layers)):
    print param_layers[num].type
  '''

  remove_ele = []
  bn_layer_location = []

  ##################################################################################################
  
  param_layers_length = len(param_layers)
  print("param_layers_length:",param_layers_length)

  i = 0
  while i < param_layers_length:
    print i
    if param_layers[i].type in layer_type:
      if (i + 2 < param_layers_length) and param_layers[i + 1].type == bnn_type[0] and param_layers[i + 2].type == bnn_type[1]:
        params = param_layers[i].param
        '''
        if len(params) < 2:
          params.add()
          params[1].lr_mult = 2 
          params[1].decay_mult = 0
          param_layers[i].convolution_param.bias_term = True
          param_layers[i].convolution_param.bias_filler.type = 'constant'
          param_layers[i].convolution_param.bias_filler.value = 0
        '''
        #modify params
        #bn_layer_location.extend([i, i + 1, i + 2])
        remove_ele.extend([param_layers[i + 1], param_layers[i + 2]])
        i = i + 3
      else:
        i += 1
    else:
      i += 1
  
  ##################################################################################################
  model_layers_length = len(model_layers)
  print("model_layers_length:",model_layers_length)  

  i = 0
  while i < model_layers_length:
    print i
    if model_layers[i].type in layer_type:
      if (i + 2 < model_layers_length) and model_layers[i + 1].type == bnn_type[0] and model_layers[i + 2].type == bnn_type[1]:

        #modify params
        bn_layer_location.extend([i, i + 1, i + 2])
        #remove_ele.extend([model_layers[i + 1], model_layers[i + 2]])
        i = i + 3
      else:
        i += 1
    else:
      i += 1

  #print bn_layer_location
  #print remove_ele
  dest_model = caffe.Net(model, caffe.TEST)
  #time.sleep(1000)
  for i, layer in enumerate(model_layers):
    if layer.type == 'Convolution' or layer.type == 'InnerProduct':
      dest_model.layers[i].blobs[0] = layer.blobs[0]
      if len(layer.blobs) > 1:
        dest_model.layers[i].blobs[1] = layer.blobs[1]

  out_model_layers = dest_model.layers
  print("model_layers:",len(model_layers))
  print("out_model_layers:",len(out_model_layers))

  l = 0
  bn_length = len(bn_layer_location)
  while l < bn_length:
    i = bn_layer_location[l]
    channels = model_layers[i].blobs[0].num
    print channels
    scale = model_layers[i + 1].blobs[2].data[0]
    #print scale
    mean = model_layers[i + 1].blobs[0].data / scale
    #print mean
    std = np.sqrt(model_layers[i + 1].blobs[1].data / scale + eps)
    a = model_layers[i + 2].blobs[0].data
    b = model_layers[i + 2].blobs[1].data
    for k in xrange(channels):
      out_model_layers[i].blobs[0].data[k] = model_layers[i].blobs[0].data[k] * a[k] / std[k] - a[k] * mean[k] / std[k] + b[k] 
      if len(model_layers[i].blobs) > 1:
        out_model_layers[i].blobs[1].data[k] = model_layers[i].blobs[1].data[k] * a[k] / std[k] - a[k] * mean[k] / std[k] + b[k] 
    l += 3
  dest_model.save(dest_weight_dir)
  out_params = param_layers
  for ele in remove_ele:
    #print ele
    out_params.remove(ele)
  
  with open(dest_model_dir, 'w') as f:
    f.write(str(net_param))
  
