import tensorflow as tf

from .heads import resnet_classification_head, resnet_scene_segmentation_head, resnet_multi_part_segmentation_head
from .backbone import resnet_backbone


class PartSegModel(object):
    def __init__(self, flat_inputs, is_training, config):
        self.config = config
        self.is_training = is_training

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['super_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['object_inds'] = flat_inputs[ind]

            self.num_layers = config.num_layers
            self.labels = self.inputs['point_labels']
            self.shape_label = self.inputs['super_labels']

        with tf.variable_scope('PartSegModel'):
            fdim = config.first_features_dim
            r = config.first_subsampling_dl * config.density_parameter
            features = self.inputs['features']
            F = resnet_backbone(config, self.inputs, features, base_radius=r, base_fdim=fdim,
                                bottleneck_ratio=config.bottleneck_ratio, depth=config.depth,
                                is_training=is_training, init=config.init, activation_fn=config.activation_fn,
                                weight_decay=config.weight_decay, bn=True, bn_momentum=config.bn_momentum,
                                bn_eps=config.bn_eps)
            self.logits_with_point_label, self.logits_all_shapes = \
                resnet_multi_part_segmentation_head(config,
                                                    self.inputs, F,
                                                    base_fdim=fdim,
                                                    is_training=is_training,
                                                    init=config.init,
                                                    weight_decay=config.weight_decay,
                                                    activation_fn=config.activation_fn,
                                                    bn=True, bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)

    def get_loss(self):
        cross_entropy = 0.0
        for i_shape in range(self.config.num_classes):
            logits_i, point_labels_i = self.logits_with_point_label[i_shape]
            cross_entropy_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=point_labels_i,
                                                                             logits=logits_i,
                                                                             name=f'cross_entropy_shape{i_shape}')
            cross_entropy += tf.reduce_sum(cross_entropy_i)

        num_inst = tf.shape(self.inputs['point_labels'])[0]
        self.loss = cross_entropy / tf.cast(num_inst, dtype=tf.float32)
        tf.add_to_collection('losses', self.loss)
        tf.add_to_collection('segmentation_losses', self.loss)


class ClassificationModel(object):
    def __init__(self, flat_inputs, is_training, config):
        self.config = config
        self.is_training = is_training

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            ind = 3 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['object_inds'] = flat_inputs[ind]

            self.num_layers = config.num_layers
            self.labels = self.inputs['labels']

        with tf.variable_scope('ClassificationModel'):
            fdim = config.first_features_dim
            r = config.first_subsampling_dl * config.density_parameter
            features = self.inputs['features']

            F = resnet_backbone(config, self.inputs, features, base_radius=r, base_fdim=fdim,
                                bottleneck_ratio=config.bottleneck_ratio, depth=config.depth,
                                is_training=is_training, init=config.init, weight_decay=config.weight_decay,
                                activation_fn=config.activation_fn, bn=True, bn_momentum=config.bn_momentum,
                                bn_eps=config.bn_eps)
            self.logits = resnet_classification_head(config, self.inputs, F[-1], base_fdim=fdim,
                                                     is_training=is_training, pooling=config.global_pooling,
                                                     init=config.init, weight_decay=config.weight_decay,
                                                     activation_fn=config.activation_fn, bn=True,
                                                     bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)

    def get_loss(self):
        labels = tf.one_hot(indices=self.labels, depth=self.config.num_classes)
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=self.logits,
                                                        label_smoothing=0.2,
                                                        scope='cross_entropy')  # be care of label smoothing

        self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        tf.add_to_collection('losses', self.loss)
        tf.add_to_collection('classification_losses', self.loss)


class SceneSegModel(object):
    def __init__(self, flat_inputs, is_training, config):
        self.config = config
        self.is_training = is_training

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_inds'] = flat_inputs[ind]
            ind += 1
            self.inputs['cloud_inds'] = flat_inputs[ind]

            self.num_layers = config.num_layers
            self.labels = self.inputs['point_labels']

        with tf.variable_scope('SceneSegModel'):
            fdim = config.first_features_dim
            r = config.first_subsampling_dl * config.density_parameter
            features = self.inputs['features']

            F = resnet_backbone(config, self.inputs, features, base_radius=r, base_fdim=fdim,
                                bottleneck_ratio=config.bottleneck_ratio, depth=config.depth,
                                is_training=is_training, init=config.init, weight_decay=config.weight_decay,
                                activation_fn=config.activation_fn, bn=True, bn_momentum=config.bn_momentum,
                                bn_eps=config.bn_eps)
            # print("F[1].shape:",F[1].shape) #F[0].shape: (?, 144)  F[1].shape: (?, 288)
            # assert 1==2

            #下面功能暂时注释
            # self.logits = resnet_scene_segmentation_head(config, self.inputs, F, base_fdim=fdim,
            #                                              is_training=is_training, init=config.init,
            #                                              weight_decay=config.weight_decay,
            #                                              activation_fn=config.activation_fn,
            #                                              bn=True, bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)
            self.logits, self.logits_contextual, self.binary, self.ori_feature, self.aug_feature = resnet_scene_segmentation_head(config, self.inputs, F, base_fdim=fdim,
                                                         is_training=is_training, init=config.init,
                                                         weight_decay=config.weight_decay,
                                                         activation_fn=config.activation_fn,
                                                         bn=True, bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)
            
            # print("self.logits.shape:",self.logits.shape) (?,20)
            # assert 1==2
            #self.logits:[num_points, num_classes]

    def get_loss(self):
        if len(self.config.ignored_label_inds) > 0:

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            # print("self.config.ignored_label_inds:",self.config.ignored_label_inds) #self.config.ignored_label_inds: [0]
            # assert 1==2
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            inds = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            new_logits = tf.gather(self.logits, inds, axis=0) #(?,20)
            new_dict = {'point_labels': tf.gather(self.labels, inds, axis=0)} #(?)
            # print("self.logits.shape:",self.logits.shape) #self.logits.shape: (?, 20)
            # print("new_logits.shape:",new_logits.shape)
            # print("new_dict['point_labels'].shape:",new_dict['point_labels'].shape)
            # assert 1==2

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32) #(0,1,...,19)
            inserted_value = tf.zeros((1,), dtype=tf.int32) #[0]
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            # print("reducing_list:",reducing_list) # Tensor("gpu_4/concat:0", shape=(21,), dtype=int32, device=/device:GPU:4)
            # print("inserted_value:",inserted_value) # Tensor("gpu_4/zeros:0", shape=(1,), dtype=int32, device=/device:GPU:4)
            # print("reducing_list:",reducing_list) # Tensor("gpu_4/concat:0", shape=(21,), dtype=int32, device=/device:GPU:4)
            # assert 1==2
            new_dict['point_labels'] = tf.gather(reducing_list, new_dict['point_labels'])

            # Add batch weigths to dict if needed
            # if self.config.batch_averaged_loss:
            #     new_dict['batch_weights'] = self.inputs['batch_weights']
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=new_dict['point_labels'],#labels=self.inputs['point_labels'],
                                                                            logits=new_logits,#logits=self.logits,
                                                                            name='cross_entropy')
            cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

            self.loss = cross_entropy
            tf.add_to_collection('losses', self.loss)
            tf.add_to_collection('segmentation_losses', self.loss)
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.inputs['point_labels'],
                                                                           logits=self.logits,
                                                                           name='cross_entropy')
            cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

            self.loss = cross_entropy
            tf.add_to_collection('losses', self.loss)
            tf.add_to_collection('segmentation_losses', self.loss)
    
    #鲁涛代码补充
    def get_contextual_loss(self):
        if len(self.config.ignored_label_inds) > 0:
    
            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))
            # Collect logits and labels that are not ignored
            inds = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            new_logits = tf.gather(self.logits_contextual, inds, axis=0) #(?,20)
            new_dict = {'point_labels': tf.gather(self.labels, inds, axis=0)} #(?)
            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32) #(0,1,...,19)
            inserted_value = tf.zeros((1,), dtype=tf.int32) #[0]
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            new_dict['point_labels'] = tf.gather(reducing_list, new_dict['point_labels'])

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=new_dict['point_labels'],
                                                                       logits=new_logits,
                                                                       name='cross_entropy_contextual')
            cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_mean_contextual')

            self.loss_contextual = cross_entropy
            tf.add_to_collection('losses_contextual', self.loss_contextual)
            tf.add_to_collection('segmentation_losses', self.loss_contextual)

        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.inputs['point_labels'],
                                                                       logits=self.logits_contextual,
                                                                       name='cross_entropy_contextual')
            cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_mean_contextual')

            self.loss_contextual = cross_entropy
            tf.add_to_collection('losses_contextual', self.loss_contextual)
            tf.add_to_collection('segmentation_losses', self.loss_contextual)
    
    #鲁涛代码补充
    def get_binary_loss(self):
        # if len(self.config.ignored_label_inds) > 0:
        
        #     # Boolean mask of points that should be ignored
        #     ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
        #     for ign_label in self.config.ignored_label_inds:
        #         ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))
        #     # Collect logits and labels that are not ignored
        #     inds = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
        #     new_logits = tf.gather(self.logits, inds, axis=0) #(?,20)
        #     new_dict = {'point_labels': tf.gather(self.labels, inds, axis=0)} #(?)
        #     # Reduce label values in the range of logit shape
        #     reducing_list = tf.range(self.config.num_classes, dtype=tf.int32) #(0,1,...,19)
        #     inserted_value = tf.zeros((1,), dtype=tf.int32) #[0]
        #     for ign_label in self.config.ignored_label_inds:
        #         reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        #     new_dict['point_labels'] = tf.gather(reducing_list, new_dict['point_labels'])

        #     neighbors = self.inputs['neighbors'][0]
        #     neighbors = neighbors[:, ::2]
        #     shadow_features = tf.concat([new_dict['point_labels'], tf.zeros_like(new_dict['point_labels'][:1])], axis=0)
        #     print("shadow_features.shape:",shadow_features.shape)
        #     print("neighbors.shape:",neighbors.shape)
        #     labels_neighbor = tf.gather(shadow_features, neighbors, axis=0)
        #     print('labels_neighbor.shape = ', labels_neighbor.shape)
        #     temp = tf.expand_dims(new_dict['point_labels'], axis=-1)

        #     n_neighbors = tf.shape(neighbors)[-1]
        #     print("neighbors.shape = ", neighbors.shape)
        #     labels_center = tf.tile(temp, [1, n_neighbors])
        #     # labels_neighbor = tf.squeeze(tf.gather(temp, neighbors), axis=-1)
        #     binary_label = tf.reshape(tf.cast(tf.equal(labels_center, labels_neighbor), tf.int32), [-1])
        #     pred_binary = tf.reshape(self.binary, [-1 ,2])
        #     print('binary_label.shape = ', binary_label.shape)
        #     print('self.binary.shape = ', self.binary.shape)
        #     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=binary_label,
        #                                                                logits=pred_binary,
        #                                                                name='cross_entropy_binary')
        #     cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_mean_binary')

        #     prediction_binary = tf.argmax(pred_binary, -1)

        #     print('prediction_binary.shape = ', prediction_binary.shape)

        #     temp = tf.cast(prediction_binary, tf.int32)==tf.cast(binary_label, tf.int32)
        #     temp = tf.cast(temp, tf.float32)
        #     self.binary_acc = tf.reduce_sum(temp)/tf.cast(tf.shape(binary_label)[0], tf.float32)

        #     self.loss_binary = cross_entropy
        #     tf.add_to_collection('losses_binary', self.loss_binary)
        # else:
            neighbors = self.inputs['neighbors'][0]
            neighbors = neighbors[:, ::2]
            shadow_features = tf.concat([self.inputs['point_labels'], tf.zeros_like(self.inputs['point_labels'][:1])], axis=0)
            labels_neighbor = tf.gather(shadow_features, neighbors, axis=0)
            print('labels_neighbor.shape = ', labels_neighbor.shape)
            temp = tf.expand_dims(self.inputs['point_labels'], axis=-1)

            n_neighbors = tf.shape(neighbors)[-1]
            print("neighbors.shape = ", neighbors.shape)
            labels_center = tf.tile(temp, [1, n_neighbors])
            # labels_neighbor = tf.squeeze(tf.gather(temp, neighbors), axis=-1)
            binary_label = tf.reshape(tf.cast(tf.equal(labels_center, labels_neighbor), tf.int32), [-1])
            pred_binary = tf.reshape(self.binary, [-1 ,2])
            print('binary_label.shape = ', binary_label.shape)
            print('self.binary.shape = ', self.binary.shape)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=binary_label,
                                                                       logits=pred_binary,
                                                                       name='cross_entropy_binary')
            cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_mean_binary')

            prediction_binary = tf.argmax(pred_binary, -1)

            print('prediction_binary.shape = ', prediction_binary.shape)

            temp = tf.cast(prediction_binary, tf.int32)==tf.cast(binary_label, tf.int32)
            temp = tf.cast(temp, tf.float32)
            self.binary_acc = tf.reduce_sum(temp)/tf.cast(tf.shape(binary_label)[0], tf.float32)

            self.loss_binary = cross_entropy
            tf.add_to_collection('losses_binary', self.loss_binary)


