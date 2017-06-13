import cv2
import numpy as np

import tensorflow as tf
from lib.roi_pooling_layer import roi_pooling_op as roi_pool_op
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.networks.VGGnet_test import VGGnet_test

#cfg_from_file('./experiments/cfgs/faster_rcnn_end2end.yml')
cfg_from_file('./experiments/cfgs/ade20k.yml')
net = VGGnet_test()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#saver = tf.train.import_meta_graph('./output/faster_rcnn_voc_vgg/ade20k/VGGnet_fast_rcnn_iter_6000.ckpt.meta')
saver = tf.train.Saver()
saver.restore(sess, './output/faster_rcnn_voc_vgg/ade20k/VGGnet_fast_rcnn_iter_6000.ckpt')
#saver.restore(sess, './output/faster_rcnn_voc_vgg/voc_2007_trainval/VGGnet_fast_rcnn_iter_15000.ckpt')

#graph = tf.get_default_graph()
#net = tf.get_collection('data')
#net = tf.get_collection('train_op')[0]

# Load the demo image
im = cv2.imread('./data/demo/ADE_train_00000037.jpg')
#im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
blobs = {'data' : None, 'rois' : None}

im_orig = im.astype(np.float32, copy=True)
im_orig -= cfg.PIXEL_MEANS

im_shape = im_orig.shape
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])

processed_ims = []
im_scale_factors = []

for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

# Create a blob to hold the input images
# blob = im_list_to_blob(processed_ims)
max_shape = np.array([im.shape]).max(axis=0)
blob = np.zeros((1, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
blob[0, 0:im.shape[0], 0:im.shape[1], :] = im

blobs['data'] = blob
im_scales = np.array(im_scale_factors)

im_blob = blobs['data']
blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

feed_dict={net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
    
cls_score, cls_prob, bbox_pred, rois = \
    sess.run([net.get_output('cls_score'), net.get_output('cls_prob'), net.get_output('bbox_pred'),net.get_output('rois')],\
    feed_dict=feed_dict)

print len(cls_prob)
#print cls_score
print cls_prob
#print bbox_pred
#print rois
