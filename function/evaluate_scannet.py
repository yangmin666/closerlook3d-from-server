"""
Evaluating script for scene segmentation with S3DIS dataset
"""
import os
import sys
import time
import pprint
import psutil
import argparse
import numpy as np
import tensorflow as tf

FILE_DIR = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(FILE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from datasets import ScannetDataset
from models import SceneSegModel
from utils.config import config, update_config
from utils.logger import setup_logger
from utils.metrics import AverageMeter,s3dis_metrics, s3dis_subset_metrics, s3dis_voting_metrics,scannet_metrics, scannet_subset_metrics, scannet_voting_metrics

def parse_option():
    parser = argparse.ArgumentParser("Evaluating ModelNet40")
    parser.add_argument('--cfg', help='yaml file', type=str)
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use [default: 0]')
    parser.add_argument('--num_threads', type=int, default=4, help='num of threads to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate for batch size 8')
    
    # IO
    parser.add_argument('--log_dir', default='log_eval', help='log dir [default: log]')
    parser.add_argument('--load_path', help='path to a check point file for load')

    # Misc
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')

    args, _ = parser.parse_known_args()

    # Update config
    update_config(args.cfg)

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    config.log_dir = os.path.join(args.log_dir, 'scannet', f'{ddir_name}_{int(time.time())}')
    config.load_path = args.load_path

    if args.num_threads:
        config.num_threads = args.num_threads
    else:
        cpu_count = psutil.cpu_count()
        config.num_threads = cpu_count
    if args.batch_size:
        config.batch_size = args.batch_size

    # Set manual seed
    tf.set_random_seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    return args, config

def evaluating(config, save_path, GPUs=0):
    logger.info("==> Start evaluating.........")
    if isinstance(GPUs, list):
        logger.warning("We use the fisrt gpu for evaluating")
        GPUs = [GPUs[0]]
    elif isinstance(GPUs, int):
        GPUs = [GPUs]
    else:
        raise RuntimeError("Check GPUs for evaluate")
    config.num_gpus = 1

    with tf.Graph().as_default():
        logger.info('==> Preparing datasets...')
        dataset = ScannetDataset(config, config.num_threads)

        #config.num_classes = dataset.num_classes #暂时注释
        config.num_classes = dataset.num_classes - len(dataset.ignored_labels)  # config.num_classes:20
        config.ignored_label_inds = [dataset.label_to_idx[ign_label] for ign_label in dataset.ignored_labels]

        flat_inputs = dataset.flat_inputs
        val_init_op = dataset.val_init_op

        is_training_pl = tf.placeholder(tf.bool, shape=())

        SceneSegModel(flat_inputs[0], is_training_pl, config=config)
        tower_logits = []

        tower_logits_contextual = [] #lutao
        tower_aug_feature = [] #lutao

        tower_labels = []
        tower_probs = []

        tower_probs_contextual = [] #lutao
        tower_binary_acc = [] #lutao

        tower_in_batches = []
        tower_point_inds = []
        tower_cloud_inds = []
        for i, igpu in enumerate(GPUs):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d' % (igpu)), tf.name_scope('gpu_%d' % (igpu)) as scope:
                    flat_inputs_i = flat_inputs[i]
                    model = SceneSegModel(flat_inputs_i, is_training_pl, config=config)
                    logits = model.logits
                    logits_contextual = model.logits_contextual #lutao
                    aug_feature = model.aug_feature #lutao
                    labels = model.labels
                    model.get_binary_loss() #lutao
                    binary_acc = model.binary_acc #lutao
                    probs = tf.nn.softmax(model.logits)
                    probs_contextual = tf.nn.softmax(model.logits_contextual) #lutao
                    tower_logits.append(logits)
                    tower_probs.append(probs)
                    tower_logits_contextual.append(logits_contextual) #lutao
                    tower_probs_contextual.append(probs_contextual) #lutao
                    tower_aug_feature.append(aug_feature) #lutao
                    tower_labels.append(labels)
                    tower_binary_acc.append(binary_acc) #lutao
                    in_batches = model.inputs['in_batches']
                    point_inds = model.inputs['point_inds']
                    cloud_inds = model.inputs['cloud_inds']
                    tower_in_batches.append(in_batches)
                    tower_point_inds.append(point_inds)
                    tower_cloud_inds.append(cloud_inds)

        # Add ops to save and restore all the variables.
        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SceneSegModel')
        saver = tf.train.Saver(save_vars)


        # Create a session
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement = True
        tfconfig.log_device_placement = False
        sess = tf.Session(config=tfconfig)

        ops = {'val_init_op': val_init_op,
               'is_training_pl': is_training_pl,
               'tower_logits': tower_logits,
               'tower_probs': tower_probs,
               'tower_logits_contextual': tower_logits_contextual, #lutao
               'tower_probs_contextual': tower_probs_contextual, #lutao
               'tower_aug_feature': tower_aug_feature, #lutao
               'tower_labels': tower_labels,
               'tower_binary_acc': tower_binary_acc, #lutao
               'tower_in_batches': tower_in_batches,
               'tower_point_inds': tower_point_inds,
               'tower_cloud_inds': tower_cloud_inds,
               }

        # Load the pretrained model
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, save_path)
        logger.info("==> Model loaded in file: %s" % save_path)

        # Evaluating
        logger.info("==> Evaluating Last epoch")
        validation_probs = [np.zeros((l.shape[0], config.num_classes)) for l in
                            dataset.input_labels['validation']]
        validation_probs_contextual = [np.zeros((l.shape[0], config.num_classes)) for l in
                            dataset.input_labels['validation']]
        # val_proportions = np.zeros(config.num_classes, dtype=np.float32)
        # for i, label_value in enumerate(dataset.label_values):
        #     val_proportions[i] = np.sum([np.sum(labels == label_value) for labels in dataset.validation_labels])
        val_proportions = np.zeros(config.num_classes, dtype=np.float32)
        i = 0
        for label_value in dataset.label_values:
            if label_value not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value) for labels in dataset.validation_labels])
                i+=1

        val_one_epoch(sess, ops, dataset, validation_probs,validation_probs_contextual, val_proportions, 'FINAL')
        val_vote_one_epoch(sess, ops, dataset, 'FINAL', num_votes=20)

    return

def val_one_epoch(sess, ops, dataset, validation_probs,validation_probs_contextual, val_proportions, epoch):
    """
    One epoch validating
    """

    is_training = False
    feed_dict = {ops['is_training_pl']: is_training}

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    val_smooth = 0.95

    # loss_meter = AverageMeter()
    # weight_loss_meter = AverageMeter()
    # seg_loss_meter = AverageMeter()

    # Initialise iterator with train data
    sess.run(ops['val_init_op'])

    idx = 0
    predictions = []
    #lutao
    predictions_contextual = []

    targets = []
    start_time = time.time() #lutao
    while True:
        try:
            tower_probs,tower_probs_contextual, tower_labels, tower_binary_acc, tower_in_batches, tower_point_inds, tower_cloud_inds = sess.run(
                [
                 ops['tower_probs'],
                 ops['tower_probs_contextual'], #lutao
                 ops['tower_labels'],
                 ops['tower_binary_acc'],
                 ops['tower_in_batches'],
                 ops['tower_point_inds'],
                 ops['tower_cloud_inds']],
                feed_dict=feed_dict)

            # print('tower_labels[0] = ', tower_labels[0].shape) #tower_labels[0] =  (89511,)
            # print('tower_probs[0].shape = ', tower_probs[0].shape) #(13170, 21) #tower_probs[0].shape =  (89511, 20)
            # print("tower_probs_contextual[0].shape = ",tower_probs_contextual[0].shape) #tower_probs_contextual[0].shape =  (89511, 20)

            # loss_meter.update(loss)
            # seg_loss_meter.update(segment_loss)
            # weight_loss_meter.update(weight_loss)

            # # Stack all validation predictions for each class separately
            # for stacked_probs, stacked_probs_contextual,labels, batches, point_inds, cloud_inds in zip(tower_probs,tower_probs_contextual, tower_labels,
            #                                                                   tower_in_batches, tower_point_inds,
            #                                                                   tower_cloud_inds):  #lutao
            #     max_ind = np.max(batches)
            #     for b_i, b in enumerate(batches):
            #         # Eliminate shadow indices
            #         b = b[b < max_ind - 0.5]
            #         # Get prediction (only for the concerned parts)
            #         probs = stacked_probs[b]
            #         probs_contextual = stacked_probs_contextual[b] #lutao
            #         # print("probs:",probs)
            #         # print("probs_contextual:",probs_contextual)
            #         inds = point_inds[b]
            #         c_i = cloud_inds[b_i]
            #         # Update current probs in whole cloud
            #         validation_probs[c_i][inds] = val_smooth * validation_probs[c_i][inds] + (1 - val_smooth) * probs
            #         validation_probs_contextual[c_i][inds] = val_smooth * validation_probs_contextual[c_i][inds] + (1 - val_smooth) * probs_contextual #lutao
            #         # Stack all prediction for this epoch
            #         predictions += [probs]
            #         predictions_contextual += [probs_contextual] #lutao
            #         targets += [dataset.input_labels['validation'][c_i][inds]]
            # if (idx + 1) % config.print_freq == 0:
            #     logger.info(f'Val: [{epoch}][{idx}] '
            #                 f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) '
            #                 f'seg loss {seg_loss_meter.val:.3f} ({seg_loss_meter.avg:.3f}) '
            #                 f'weight loss {weight_loss_meter.val:.3f} ({weight_loss_meter.avg:.3f})')
            idx += 1
        except tf.errors.OutOfRangeError:
            break
    
    end_time = time.time() #lutao
    print('time cost = ', end_time-start_time) #lutao
    #IoUs, mIoU = s3dis_subset_metrics(dataset, predictions, targets, val_proportions)
    # print("predictions[0].shape:",predictions[0].shape) #(9813, 21)   #predictions[0].shape: (10486, 20)
    # print("predictions_contextual[0].shape:",predictions_contextual[0].shape) #predictions_contextual[0].shape: (10486, 20)
    IoUs, mIoU = scannet_subset_metrics(dataset, predictions, targets, val_proportions)
    #vote_IoUs, vote_mIoU = s3dis_voting_metrics(dataset, validation_probs, val_proportions)
    vote_IoUs, vote_mIoU = scannet_voting_metrics(dataset, validation_probs, val_proportions)

    IoUs_contextual, mIoU_contextual = scannet_subset_metrics(dataset, predictions_contextual, targets, val_proportions) #lutao
    vote_IoUs_contextual, vote_mIoU_contextual = scannet_voting_metrics(dataset, validation_probs_contextual, val_proportions) #lutao


    logger.info(f'E{epoch} * mIoU {mIoU:.3%} vote_mIoU {vote_mIoU:.3%}')
    logger.info(f'E{epoch} * IoUs {IoUs}')
    logger.info(f'E{epoch} * vote_IoUs {vote_IoUs}')

    logger.info(f'E{epoch} * mIoU contextual {mIoU_contextual:.3%} vote_mIoU contextual {vote_mIoU_contextual:.3%}')
    logger.info(f'E{epoch} * IoUs contextual {IoUs_contextual}')
    logger.info(f'E{epoch} * vote_IoUs contextual {vote_IoUs_contextual}')

    return


def val_vote_one_epoch(sess, ops, dataset, epoch, num_votes=20):
    """
    One epoch voting validating
    """

    is_training = False
    feed_dict = {ops['is_training_pl']: is_training}

    # Smoothing parameter for votes
    test_smooth = 0.95

    # Initialise iterator with val data
    sess.run(ops['val_init_op'])

    #下面两行是我加的
    # config.num_classes = dataset.num_classes - len(dataset.ignored_labels)  # config.num_classes:20
    # config.ignored_label_inds = [dataset.label_to_idx[ign_label] for ign_label in dataset.ignored_labels]

    print("dataset.num_classes:",dataset.num_classes) #dataset.num_classes: 21
    print("config.num_classes:",config.num_classes) #config.num_classes: 20
    # assert 1==2

    # Initiate global prediction over test clouds
    #暂时注释下面两行源码
    # nc_model = dataset.num_classes #21  
    # val_probs = [np.zeros((l.shape[0], nc_model), dtype=np.float32) for l in dataset.input_labels['validation']]

    val_probs = [np.zeros((l.shape[0],config.num_classes), dtype=np.float32) for l in dataset.input_labels['validation']]

    #暂时注释下面三行源码
    # val_proportions = np.zeros(nc_model, dtype=np.float32)
    # for i, label_value in enumerate(dataset.label_values):
    #     val_proportions[i] = np.sum([np.sum(labels == label_value) for labels in dataset.validation_labels])
    val_probs_contextual = [np.zeros((l.shape[0], config.num_classes), dtype=np.float32) for l in dataset.input_labels['validation']]
    val_proportions = np.zeros(config.num_classes, dtype=np.float32)
    i = 0
    for label_value in dataset.label_values:
        if label_value not in dataset.ignored_labels:
            val_proportions[i] = np.sum([np.sum(labels == label_value) for labels in dataset.validation_labels])
            i+=1

    vote_ind = 0
    last_min = -0.5
    while last_min < num_votes:
        try:
            tower_probs,tower_probs_contextual, tower_labels, tower_in_batches, tower_point_inds, tower_cloud_inds = sess.run(
                [ops['tower_probs'],
                 ops['tower_probs_contextual'], #lutao
                 ops['tower_labels'],
                 ops['tower_in_batches'],
                 ops['tower_point_inds'],
                 ops['tower_cloud_inds']],
                feed_dict=feed_dict)
            for stacked_probs,stacked_probs_contextual,labels, batches, point_inds, cloud_inds in zip(tower_probs,tower_probs_contextual, tower_labels,
                                                                              tower_in_batches, tower_point_inds,
                                                                              tower_cloud_inds):
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    probs_contextual = stacked_probs_contextual[b] #lutao
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    val_probs[c_i][inds] = test_smooth * val_probs[c_i][inds] + (1 - test_smooth) * probs
                    val_probs_contextual[c_i][inds] = test_smooth * val_probs_contextual[c_i][inds] + (1 - test_smooth) * probs_contextual
        except:
            new_min = np.min(dataset.min_potentials['validation'])
            logger.info('Step {:3d}, end. Min potential = {:.1f}'.format(vote_ind, new_min))
            if last_min + 1 < new_min:
                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                logger.info('==> Confusion on sub clouds')
                #IoUs, mIoU = s3dis_voting_metrics(dataset, val_probs, val_proportions)
                IoUs, mIoU = scannet_voting_metrics(dataset, val_probs, val_proportions)
                IoUs_contextual, mIoU_contextual = scannet_voting_metrics(dataset, val_probs_contextual, val_proportions) #lutao
                logger.info(f'E{epoch} S{vote_ind} * mIoU {mIoU:.3%}')
                logger.info(f'E{epoch} S{vote_ind} * mIoU_contextual {mIoU_contextual:.3%}') #lutao

                if int(np.ceil(new_min)) % 2 == 0:
                    # Project predictions
                    v = int(np.floor(new_min))
                    logger.info('Reproject True Vote #{:d}'.format(v))
                    files = dataset.train_files
                    i_val = 0
                    proj_probs = []
                    proj_probs_contextual = [] #lutao
                    for i, file_path in enumerate(files):
                        if dataset.all_splits[i] == dataset.validation_split:
                            # Reproject probs on the evaluations points
                            probs = val_probs[i_val][dataset.validation_proj[i_val], :]
                            proj_probs += [probs]
                            #lutao 两行
                            probs_contextual = val_probs_contextual[i_val][dataset.validation_proj[i_val], :]
                            proj_probs_contextual += [probs_contextual]

                            i_val += 1
                    # Show vote results
                    logger.info('==> Confusion on full clouds')
                    #IoUs, mIoU = s3dis_metrics(dataset, proj_probs)
                    IoUs, mIoU = scannet_metrics(dataset, proj_probs)
                    IoUs_contextual, mIoU_contextual = scannet_metrics(dataset, proj_probs_contextual) #lutao
                    logger.info(f'E{epoch} V{v} * mIoU {mIoU:.3%}')
                    logger.info(f'E{epoch} V{v} * IoUs {IoUs}')

                    #lutao 两行
                    logger.info(f'E{epoch} V{v} * mIoU_contextual {mIoU_contextual:.3%}')
                    logger.info(f'E{epoch} V{v} * IoUs_contextual {IoUs_contextual}')

            sess.run(ops['val_init_op'])
            vote_ind += 1

    # Project predictions
    logger.info('Reproject True Vote Last')
    files = dataset.train_files
    i_val = 0
    proj_probs = []
    for i, file_path in enumerate(files):
        if dataset.all_splits[i] == dataset.validation_split:
            # Reproject probs on the evaluations points
            probs = val_probs[i_val][dataset.validation_proj[i_val], :]
            proj_probs += [probs]

            #lutao 两行
            probs_contextual = val_probs_contextual[i_val][dataset.validation_proj[i_val], :]
            proj_probs_contextual += [probs_contextual]

            i_val += 1
       # print('file_path = ', file_path)
    
    # Show vote results
    # print('proj_probs_contextual[0].shape = ', proj_probs_contextual[0].shape)

    # preds = dataset.label_values[np.argmax(proj_probs[0], axis=1)].astype(np.uint8)
    # preds_contextual = dataset.label_values[np.argmax(proj_probs_contextual[0], axis=1)].astype(np.uint8)

    # print('proj_probs[0].shape = ', proj_probs[0].shape)
    # print('proj_probs_contextual[0].shape = ', proj_probs_contextual[0].shape)

    # labels = dataset.validation_labels[0]
    # name='area5.ply'
    # np.savez(os.path.join('val_preds', name), pred=preds, probs=proj_probs[0], label=labels)
    # np.savez(os.path.join('val_preds_contextual', name), pred=preds_contextual, probs=proj_probs_contextual[0], label=labels)

    # print('preds.shape = ', preds.shape)
    # print('preds_contextual.shape = ', preds_contextual.shape)
    # Show vote results
    logger.info('==> Confusion on full clouds')
    #IoUs, mIoU = s3dis_metrics(dataset, proj_probs)
    IoUs, mIoU = scannet_metrics(dataset, proj_probs)
    logger.info(f'E{epoch} VLast * mIoU {mIoU:.3%}')
    logger.info(f'E{epoch} VLast * IoUs {IoUs}')
    #lutao 两行
    #IoUs_contextual, mIoU_contextual = scannet_metrics(dataset, proj_probs_contextual)
    logger.info(f'E{epoch} VLast * mIoU_contextual {mIoU_contextual:.3%}')
    logger.info(f'E{epoch} VLast * IoUs_contextual {IoUs_contextual}')

    return

if __name__ == "__main__":
    args, config = parse_option()
    os.makedirs(config.log_dir, exist_ok=True)
    logger = setup_logger(output=config.log_dir, name="scannet_eval")
    logger.info(pprint.pformat(config))
    evaluating(config, config.load_path, args.gpu)