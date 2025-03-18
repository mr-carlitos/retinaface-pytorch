##### CARLOS CODE FILE ########

import mxnet as mx
import numpy as np
from config import config
from config import default
import argparse


# Paths to your model files
symbol_file = 'resnet-50-symbol.json'
params_file = 'resnet-50-0000.params'

# Load the network symbol
sym = mx.sym.load(symbol_file)


PREFIX = 'RF'
F1 = 0
F2 = 0
_bwm = 1.0

def conv_only(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), bias_wd_mult=0.0, shared_weight=None, shared_bias = None):
    if shared_weight is None:
        weight = mx.symbol.Variable(name="{}_weight".format(name),
                                    init=mx.init.Normal(0.01),
                                    attr={'__lr_mult__': '1.0'})
        bias = mx.symbol.Variable(name="{}_bias".format(name),
                                  init=mx.init.Constant(0.0),
                                  attr={
                                      '__lr_mult__': '2.0',
                                      '__wd_mult__': str(bias_wd_mult)
                                  })
    else:
        weight = shared_weight
        bias = shared_bias
        print('reuse shared var in', name)
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}".format(name), weight = weight, bias=bias)
    return conv


def conv_deformable(net, num_filter, num_group=1, act_type='relu', name=''):
    if config.USE_DCN == 1:
        f = num_group * 18
        conv_offset = mx.symbol.Convolution(name=name + '_conv_offset',
                                            data=net,
                                            num_filter=f,
                                            pad=(1, 1),
                                            kernel=(3, 3),
                                            stride=(1, 1))
        net = mx.contrib.symbol.DeformableConvolution(
            name=name + "_conv",
            data=net,
            offset=conv_offset,
            num_filter=num_filter,
            pad=(1, 1),
            kernel=(3, 3),
            num_deformable_group=num_group,
            stride=(1, 1),
            no_bias=False)
    else:
        print('use dcnv2 at', name)
        lr_mult = 0.1
        weight_var = mx.sym.Variable(name=name + '_conv2_offset_weight',
                                     init=mx.init.Zero(),
                                     lr_mult=lr_mult)
        bias_var = mx.sym.Variable(name=name + '_conv2_offset_bias',
                                   init=mx.init.Zero(),
                                   lr_mult=lr_mult)
        conv2_offset = mx.symbol.Convolution(name=name + '_conv2_offset',
                                             data=net,
                                             num_filter=27,
                                             pad=(1, 1),
                                             kernel=(3, 3),
                                             stride=(1, 1),
                                             weight=weight_var,
                                             bias=bias_var,
                                             lr_mult=lr_mult)
        conv2_offset_t = mx.sym.slice_axis(conv2_offset,
                                           axis=1,
                                           begin=0,
                                           end=18)
        conv2_mask = mx.sym.slice_axis(conv2_offset,
                                       axis=1,
                                       begin=18,
                                       end=None)
        conv2_mask = 2 * mx.sym.Activation(conv2_mask, act_type='sigmoid')

        conv2 = mx.contrib.symbol.ModulatedDeformableConvolution(
            name=name + '_conv2',
            data=net,
            offset=conv2_offset_t,
            mask=conv2_mask,
            num_filter=num_filter,
            pad=(1, 1),
            kernel=(3, 3),
            stride=(1, 1),
            num_deformable_group=num_group,
            no_bias=True)
        net = conv2
    net = mx.sym.BatchNorm(data=net,
                           fix_gamma=False,
                           eps=2e-5,
                           momentum=0.9,
                           name=name + '_bn')
    if len(act_type) > 0:
        net = mx.symbol.Activation(data=net,
                                   act_type=act_type,
                                   name=name + '_act')
    return net

def conv_act_layer_dw(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", bias_wd_mult=0.0):
    assert kernel[0] == 3
    weight = mx.symbol.Variable(name="{}_weight".format(name),
                                init=mx.init.Normal(0.01),
                                attr={'__lr_mult__': '1.0'})
    bias = mx.symbol.Variable(name="{}_bias".format(name),
                              init=mx.init.Constant(0.0),
                              attr={
                                  '__lr_mult__': '2.0',
                                  '__wd_mult__': str(bias_wd_mult)
                              })
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, num_group=num_filter, name="{}".format(name), weight=weight, bias=bias)
    conv = mx.sym.BatchNorm(data=conv,
                            fix_gamma=False,
                            eps=2e-5,
                            momentum=0.9,
                            name=name + '_bn')
    if len(act_type) > 0:
        relu = mx.symbol.Activation(data=conv, act_type=act_type, \
            name="{}_{}".format(name, act_type))
    else:
        relu = conv
    return relu

def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", bias_wd_mult=0.0, separable=False, filter_in = -1):

    if config.USE_DCN > 1 and kernel == (3, 3) and pad == (
            1, 1) and stride == (1, 1) and not separable:
        return conv_deformable(from_layer,
                               num_filter,
                               num_group=1,
                               act_type=act_type,
                               name=name)

    if separable:
        assert kernel[0] > 1
        assert filter_in > 0
    if not separable:
        weight = mx.symbol.Variable(name="{}_weight".format(name),
                                    init=mx.init.Normal(0.01),
                                    attr={'__lr_mult__': '1.0'})
        bias = mx.symbol.Variable(name="{}_bias".format(name),
                                  init=mx.init.Constant(0.0),
                                  attr={
                                      '__lr_mult__': '2.0',
                                      '__wd_mult__': str(bias_wd_mult)
                                  })
        conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
            stride=stride, num_filter=num_filter, name="{}".format(name), weight=weight, bias=bias)
        conv = mx.sym.BatchNorm(data=conv,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=0.9,
                                name=name + '_bn')
    else:
        if filter_in < 0:
            filter_in = num_filter
        conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
            stride=stride, num_filter=filter_in, num_group=filter_in, name="{}_sep".format(name))
        conv = mx.sym.BatchNorm(data=conv,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=0.9,
                                name=name + '_sep_bn')
        conv = mx.symbol.Activation(data=conv, act_type='relu', \
            name="{}_sep_bn_relu".format(name))
        conv = mx.symbol.Convolution(data=conv, kernel=(1,1), pad=(0,0), \
            stride=(1,1), num_filter=num_filter, name="{}".format(name))
        conv = mx.sym.BatchNorm(data=conv,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=0.9,
                                name=name + '_bn')
    if len(act_type) > 0:
        relu = mx.symbol.Activation(data=conv, act_type=act_type, \
            name="{}_{}".format(name, act_type))
    else:
        relu = conv
    return relu


def ssh_context_module(body, num_filter, filter_in, name):
    conv_dimred = conv_act_layer(body,
                                 name + '_conv1',
                                 num_filter,
                                 kernel=(3, 3),
                                 pad=(1, 1),
                                 stride=(1, 1),
                                 act_type='relu',
                                 separable=False,
                                 filter_in=filter_in)
    conv5x5 = conv_act_layer(conv_dimred,
                             name + '_conv2',
                             num_filter,
                             kernel=(3, 3),
                             pad=(1, 1),
                             stride=(1, 1),
                             act_type='',
                             separable=False)
    conv7x7_1 = conv_act_layer(conv_dimred,
                               name + '_conv3_1',
                               num_filter,
                               kernel=(3, 3),
                               pad=(1, 1),
                               stride=(1, 1),
                               act_type='relu',
                               separable=False)
    conv7x7 = conv_act_layer(conv7x7_1,
                             name + '_conv3_2',
                             num_filter,
                             kernel=(3, 3),
                             pad=(1, 1),
                             stride=(1, 1),
                             act_type='',
                             separable=False)
    return (conv5x5, conv7x7)


def ssh_detection_module(body, num_filter, filter_in, name):
    assert num_filter % 4 == 0
    conv3x3 = conv_act_layer(body,
                             name + '_conv1',
                             num_filter // 2,
                             kernel=(3, 3),
                             pad=(1, 1),
                             stride=(1, 1),
                             act_type='',
                             separable=False,
                             filter_in=filter_in)
    #_filter = max(num_filter//4, 16)
    _filter = num_filter // 4
    conv5x5, conv7x7 = ssh_context_module(body, _filter, filter_in,
                                          name + '_context')
    ret = mx.sym.concat(*[conv3x3, conv5x5, conv7x7],
                        dim=1,
                        name=name + '_concat')
    ret = mx.symbol.Activation(data=ret,
                               act_type='relu',
                               name=name + '_concat_relu')
    out_filter = num_filter // 2 + _filter * 2
    if config.USE_DCN > 0:
        ret = conv_deformable(ret,
                              num_filter=out_filter,
                              name=name + '_concat_dcn')
    return ret


#def retina_context_module(body, kernel, num_filter, filter_in, name):
#  conv_dimred = conv_act_layer(body, name+'_conv0',
#      num_filter, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu', separable=False, filter_in = filter_in)
#  conv1 = conv_act_layer(conv_dimred, name+'_conv1',
#      num_filter*6, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu', separable=False, filter_in = filter_in)
#  conv2 = conv_act_layer(conv1, name+'_conv2',
#      num_filter*6, kernel=kernel, pad=((kernel[0]-1)//2, (kernel[1]-1)//2), stride=(1, 1), act_type='relu', separable=True, filter_in = num_filter*6)
#  conv3 = conv_act_layer(conv2, name+'_conv3',
#      num_filter, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu', separable=False)
#  conv3 = conv3 + conv_dimred
#  return conv3


def retina_detection_module(body, num_filter, filter_in, name):
    assert num_filter % 4 == 0
    conv1 = conv_act_layer(body,
                           name + '_conv1',
                           num_filter // 2,
                           kernel=(3, 3),
                           pad=(1, 1),
                           stride=(1, 1),
                           act_type='relu',
                           separable=False,
                           filter_in=filter_in)
    conv2 = conv_act_layer(conv1,
                           name + '_conv2',
                           num_filter // 2,
                           kernel=(3, 3),
                           pad=(1, 1),
                           stride=(1, 1),
                           act_type='relu',
                           separable=False,
                           filter_in=num_filter // 2)
    conv3 = conv_act_layer(conv2,
                           name + '_conv3',
                           num_filter // 2,
                           kernel=(3, 3),
                           pad=(1, 1),
                           stride=(1, 1),
                           act_type='relu',
                           separable=False,
                           filter_in=num_filter // 2)
    conv4 = conv2 + conv3
    body = mx.sym.concat(*[conv1, conv4], dim=1, name=name + '_concat')
    if config.USE_DCN > 0:
        body = conv_deformable(body,
                               num_filter=num_filter,
                               name=name + '_concat_dcn')
    return body


def head_module(body, num_filter, filter_in, name):
    if config.HEAD_MODULE == 'SSH':
        return ssh_detection_module(body, num_filter, filter_in, name)
    else:
        return retina_detection_module(body, num_filter, filter_in, name)


def upsampling(data, num_filter, name):
    #ret = mx.symbol.Deconvolution(data=data, num_filter=num_filter, kernel=(4,4),  stride=(2, 2), pad=(1,1),
    #    num_group = num_filter, no_bias = True, attr={'__lr_mult__': '0.0', '__wd_mult__': '0.0'},
    #    name=name)
    #ret = mx.symbol.Deconvolution(data=data, num_filter=num_filter, kernel=(2,2),  stride=(2, 2), pad=(0,0),
    #    num_group = num_filter, no_bias = True, attr={'__lr_mult__': '0.0', '__wd_mult__': '0.0'},
    #    name=name)
    ret = mx.symbol.UpSampling(data,
                               scale=2,
                               sample_type='nearest',
                               workspace=512,
                               name=name,
                               num_args=1)
    return ret


def get_sym_by_name(name, sym_buffer):
    if name in sym_buffer:
        return sym_buffer[name]
    ret = None
    name_key = name[0:1]
    name_num = int(name[1:])
    #print('getting', name, name_key, name_num)
    if name_key == 'C':
        assert name_num % 2 == 0
        bottom = get_sym_by_name('C%d' % (name_num // 2), sym_buffer)
        ret = conv_act_layer(bottom,
                             '%s_C%d' % (PREFIX, name_num),
                             F1,
                             kernel=(3, 3),
                             pad=(1, 1),
                             stride=(2, 2),
                             act_type='relu',
                             bias_wd_mult=_bwm)
    elif name_key == 'P':
        assert name_num % 2 == 0
        assert name_num <= max([32, 16, 8])
        lateral = get_sym_by_name('L%d' % (name_num), sym_buffer)
        if name_num == max([32, 16, 8]) or name_num > 32:
            ret = mx.sym.identity(lateral, name='%s_P%d' % (PREFIX, name_num))
        else:
            bottom = get_sym_by_name('L%d' % (name_num * 2), sym_buffer)
            bottom_up = upsampling(bottom, F1, '%s_U%d' % (PREFIX, name_num))
            if True:
                bottom_up = mx.symbol.Crop(*[bottom_up, lateral])
            aggr = lateral + bottom_up
            aggr = conv_act_layer(aggr,
                                  '%s_A%d' % (PREFIX, name_num),
                                  F1,
                                  kernel=(3, 3),
                                  pad=(1, 1),
                                  stride=(1, 1),
                                  act_type='relu',
                                  bias_wd_mult=_bwm)
            ret = mx.sym.identity(aggr, name='%s_P%d' % (PREFIX, name_num))
    elif name_key == 'L':
        c = get_sym_by_name('C%d' % (name_num), sym_buffer)
        #print('L', name, F1)
        ret = conv_act_layer(c,
                             '%s_L%d' % (PREFIX, name_num),
                             F1,
                             kernel=(1, 1),
                             pad=(0, 0),
                             stride=(1, 1),
                             act_type='relu',
                             bias_wd_mult=_bwm)
    else:
        raise RuntimeError('%s is not a valid sym key name' % name)
    sym_buffer[name] = ret
    return ret

def get_sym_conv(sym):
    all_layers = sym.get_internals()
    isize = 640
    _, out_shape, _ = all_layers.infer_shape(data=(1, 3, isize, isize))
    outputs = all_layers.list_outputs()
    count = len(outputs)
    stride2name = {}
    stride2layer = {}
    stride2shape = {}
    for i in range(count):
        name = outputs[i]
        shape = out_shape[i]
        print(i, name, count, shape)
        if not name.endswith('_output'):
            continue
        if len(shape) != 4:
            continue
        assert isize % shape[2] == 0
        if shape[1] > 9999:
            break
        stride = isize // shape[2]
        stride2name[stride] = name
        stride2layer[stride] = all_layers[name]
        stride2shape[stride] = shape
    strides = sorted(stride2name.keys())
    for stride in strides:
        print('stride', stride, stride2name[stride], stride2shape[stride])

    for stride in strides:
        print('stride', stride, stride2name[stride], stride2shape[stride])
    _bwm = 1.0
    ret = {}
    sym_buffer = {}
    for stride in [4, 8, 16, 32]:
        sym_buffer['C%d' % stride] = stride2layer[stride]
    for stride in [32, 16, 8]:
        name = 'P%d' % stride
        ret[stride] = get_sym_by_name(name, sym_buffer)

    return ret


def get_sym_train(sym):
    data = mx.symbol.Variable(name="data")
    global F1, F2
    F1 = config.HEAD_FILTER_NUM
    F2 = F1

    # shared convolutional layers
    conv_fpn_feat = get_sym_conv(sym)
    internals = conv_fpn_feat[8].get_internals()
    internals2 = conv_fpn_feat[16].get_internals()
    internals3 = conv_fpn_feat[32].get_internals()
    print(internals.list_outputs())
    print(internals2.list_outputs())
    print(internals3.list_outputs())


get_sym_train(sym)