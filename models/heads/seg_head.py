import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(ROOT_DIR)

from ..local_aggregation_operators import *


def nearest_upsample_block(layer_ind, inputs, features, scope):
    """
    This Block performing an upsampling by nearest interpolation
    Args:
        layer_ind: Upsampled to which layer
        inputs: a dict contains all inputs
        features: x = [n1, d]
        scope: name scope

    Returns:
        x = [n2, d]
    """

    with tf.variable_scope(scope) as sc:
        upsampled_features = ind_closest_pool(features, inputs['upsamples'][layer_ind], 'nearest_upsample')
        return upsampled_features


def resnet_multi_part_segmentation_head(config,
                                        inputs,
                                        F,
                                        base_fdim,
                                        is_training,
                                        init='xavier',
                                        weight_decay=0,
                                        activation_fn='relu',
                                        bn=True,
                                        bn_momentum=0.98,
                                        bn_eps=1e-3):
    """A head for multi-shape part segmentation with resnet backbone.

    Args:
        config: config file
        inputs: a dict contains all inputs
        F: all stage features
        base_fdim: the base feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        logits for all shapes with all parts  [num_classes, num_points, num_parts_i]
    """
    with tf.variable_scope('resnet_multi_part_segmentation_head') as sc:
        fdim = base_fdim
        features = F[-1]

        features = nearest_upsample_block(4, inputs, features, 'nearest_upsample_0')
        features = tf.concat((features, F[3]), axis=1)
        features = conv1d_1x1(features, 8 * fdim, 'up_conv0', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(3, inputs, features, 'nearest_upsample_1')
        features = tf.concat((features, F[2]), axis=1)
        features = conv1d_1x1(features, 4 * fdim, 'up_conv1', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(2, inputs, features, 'nearest_upsample_2')
        features = tf.concat((features, F[1]), axis=1)
        features = conv1d_1x1(features, 2 * fdim, 'up_conv2', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(1, inputs, features, 'nearest_upsample_3')
        features = tf.concat((features, F[0]), axis=1)
        features = conv1d_1x1(features, fdim, 'up_conv3', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        shape_heads = []
        for i_shape in range(config.num_classes):
            head = features
            head = conv1d_1x1(head, fdim, f'shape{i_shape}_head', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

            head = conv1d_1x1(head, config.num_parts[i_shape], f'shape{i_shape}_pred', is_training=is_training,
                              with_bias=True, init=init,
                              weight_decay=weight_decay, activation_fn=None, bn=False)
            shape_heads.append(head)

        shape_label = inputs['super_labels']
        logits_with_point_label = [()] * config.num_classes
        for i_shape in range(config.num_classes):
            i_shape_inds = tf.where(tf.equal(shape_label, i_shape))
            logits_i = tf.gather_nd(shape_heads[i_shape], i_shape_inds)
            point_labels_i = tf.gather_nd(inputs['point_labels'], i_shape_inds)
            logits_with_point_label[i_shape] = (logits_i, point_labels_i)
        logits_all_shapes = shape_heads

    return logits_with_point_label, logits_all_shapes


def resnet_scene_segmentation_head(config,
                                   inputs,
                                   F,
                                   base_fdim,
                                   is_training,
                                   init='xavier',
                                   weight_decay=0,
                                   activation_fn='relu',
                                   bn=True,
                                   bn_momentum=0.98,
                                   bn_eps=1e-3):
    """A head for scene segmentation with resnet backbone.

    Args:
        config: config file
        inputs: a dict contains all inputs
        F: all stage features
        base_fdim: the base feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        prediction logits [num_points, num_classes]
    """
    with tf.variable_scope('resnet_scene_segmentation_head') as sc:
        fdim = base_fdim
        features = F[-1]

        features = nearest_upsample_block(4, inputs, features, 'nearest_upsample_0')
        features = tf.concat((features, F[3]), axis=1)
        features = conv1d_1x1(features, 8 * fdim, 'up_conv0', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(3, inputs, features, 'nearest_upsample_1')
        features = tf.concat((features, F[2]), axis=1)
        features = conv1d_1x1(features, 4 * fdim, 'up_conv1', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(2, inputs, features, 'nearest_upsample_2')
        features = tf.concat((features, F[1]), axis=1)
        features = conv1d_1x1(features, 2 * fdim, 'up_conv2', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(1, inputs, features, 'nearest_upsample_3')
        features = tf.concat((features, F[0]), axis=1)
        features = conv1d_1x1(features, fdim, 'up_conv3', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = conv1d_1x1(features, fdim, 'segmentation_head', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        # print("@@@config.num_classes:",config.num_classes)
        # assert 1==2
        logits = conv1d_1x1(features, config.num_classes, 'segmentation_pred', is_training=is_training, with_bias=True,
                            init=init, weight_decay=weight_decay, activation_fn=None, bn=False)
        #(?,20)

        #lutao
        f_dim = features.get_shape()[-1].value
        with tf.variable_scope('refine', reuse=tf.AUTO_REUSE):
            to_be_augmented = features
            binary, intra_features, inter_features = feature_augmentation(inputs['points'][0],
                                                                          inputs['neighbors'][0],
                                                                          inputs['point_labels'],
                                                                          to_be_augmented,
                                                                            init='xavier',
                                                                            is_training=is_training,
                                                                            weight_decay=weight_decay,
                                                                            activation_fn='relu',
                                                                            bn=True,
                                                                            bn_momentum=0.98,
                                                                            bn_eps=1e-3)
            net_intra = tf.concat([to_be_augmented, intra_features], axis=-1)
            net_intra = conv1d_1x1(net_intra, 32, 'intra_fc2', is_training=is_training, with_bias=False, init=init,
                                   weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                   bn_momentum=bn_momentum, bn_eps=bn_eps)
            net_inter = tf.concat([to_be_augmented, inter_features], axis=-1)
            net_inter = conv1d_1x1(net_inter, 32, 'inter_fc2', is_training=is_training, with_bias=False, init=init,
                                   weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                   bn_momentum=bn_momentum, bn_eps=bn_eps)

            contextual_concat = tf.concat([to_be_augmented, net_intra, net_inter], axis=-1)
            contextual_feature = conv1d_1x1(contextual_concat, f_dim, 'contextual_fc2', is_training=is_training, with_bias=False, init=init,
                                   weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                   bn_momentum=bn_momentum, bn_eps=bn_eps)
            per_feature = contextual_feature
            f_layer_contextual_fc = conv1d_1x1(contextual_feature, config.num_classes, 'fc1_contextual', is_training=is_training, with_bias=False, init=init,
                                   weight_decay=weight_decay, activation_fn=None, bn=bn,
                                   bn_momentum=bn_momentum, bn_eps=bn_eps)
            logits_contextual = f_layer_contextual_fc
    
    ori_feature = to_be_augmented
    aug_feature = per_feature
    return logits, logits_contextual, binary, ori_feature, aug_feature
    #return logits

def feature_augmentation(xyz, neighbor_idx, labels, F, init, is_training, weight_decay, activation_fn,bn,bn_momentum,bn_eps):
    """A head for scene segmentation with resnet backbone.
        Args:
            xyz: BxNx3
            neighbor_idx: Nxn_neighbor
            F: BxNx1xC
            base_fdim: the base feature dim
            is_training: True indicates training phase
        Returns:
            prediction logits [num_points, num_classes]
        """
    neighbor_idx = neighbor_idx[:,::2]
    with tf.variable_scope('feature_augmentation_head') as sc:
        print('xyz.shape = ', xyz.shape)
        N = tf.shape(xyz)[0]
        fdim = 32#tf.shape(F)[-1]#F.shape[-1]
        features = F

        query_points = xyz
        # neighbor_idx = neighbor_idx[:,:,::4]
        print('aug, features.shape = ', features.shape)
        # features = tf.squeeze(features, axis=1)
        neighbor_features = tf.gather(features, neighbor_idx, axis=0)
        features = tf.expand_dims(features, axis=1)
        features = tf.tile(features, [1, tf.shape(neighbor_idx)[-1], 1])
        center_points = tf.expand_dims(query_points, 1)
        neighbor_points = tf.gather(query_points, neighbor_idx, axis=0)
        diff_xyz = center_points - neighbor_points

        concat_features_xyz = tf.concat([features, neighbor_features, diff_xyz], axis=-1)
        concat_features_xyz = tf.reshape(concat_features_xyz, [tf.shape(xyz)[0]*tf.shape(neighbor_idx)[-1], 2*features.get_shape()[-1].value+3])
        # concat_features_xyz = tf.expand_dims(concat_features_xyz, axis=2)
        binary = conv1d_1x1(concat_features_xyz, 2, 'binary_pred', is_training=is_training, with_bias=False, init=init,
                                   weight_decay=weight_decay, activation_fn=None, bn=bn,
                                   bn_momentum=bn_momentum, bn_eps=bn_eps)
        # binary = helper_tf_util.conv2d(concat_features_xyz, 2, [1, 1], 'binary_pred', [1, 1], 'VALID', True, is_training)
        binary_temp = binary
        binary_temp = tf.reshape(binary_temp, [tf.shape(xyz)[0], tf.shape(neighbor_idx)[-1], 2])
        binary_soft = tf.nn.softmax(binary, axis=-1)
        binary_soft = tf.reshape(binary_soft, [tf.shape(xyz)[0], tf.shape(neighbor_idx)[-1], 2])
        binary_soft = tf.ones_like(binary_soft)
        intra_neighbor = binary_soft[:, :, 1:]
        inter_neighbor = binary_soft[:, :, :1]

        intra_features = tf.multiply(intra_neighbor, neighbor_features)
        intra_features = tf.div_no_nan(tf.reduce_sum(intra_features, axis=-2),tf.reduce_sum(intra_neighbor, axis=-2),)

        concat_features_diff = tf.concat([features - neighbor_features, diff_xyz], axis=-1)
        concat_features_diff = tf.reshape(concat_features_diff, [N*tf.shape(neighbor_idx)[-1], features.get_shape()[-1].value+3])

        rel_feature = conv1d_1x1(concat_features_diff, fdim, 'aug_diff_feature', is_training=is_training, with_bias=False, init=init,
                                   weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                   bn_momentum=bn_momentum, bn_eps=bn_eps)
        # aug_feature = helper_tf_util.conv2d_relu(concat_features_diff, fdim, [1, 1], 'aug_diff_feature', [1, 1], 'VALID', True, is_training)
        rel_feature = tf.reshape(rel_feature, [N, tf.shape(neighbor_idx)[-1], fdim])
        inter_features = tf.multiply(inter_neighbor, rel_feature)
        inter_features = tf.div_no_nan(tf.reduce_sum(inter_features, axis=-2),
                                       tf.reduce_sum(inter_neighbor, axis=-2) + 1e-10)
        # inter_features = tf.expand_dims(inter_features, axis=-2)
        print('intra_features.shape, inter_features.shape = ', intra_features.shape, inter_features.shape)



    return binary_temp, intra_features, inter_features