import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.utils.blob import im_list_to_blob
from lib.fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

'''
'''
# ade20k
CLASSES = ('__background__', # always index 0
           'door')

# progress-obj
'''
CLASSES = ('__background__', # always index 0
           'tide', 'spray_bottle_a', 'waterpot', 'sugar',
           'red_bowl', 'clorox', 'shampoo', 'downy', 'salt',
           'toy', 'detergent', 'scotch_brite', 'cola',
           'blue_cup', 'ranch')
'''

'''
# VOC
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
'''

'''
# coco
CLASSES = ('__background__','person','bicycle','car','motorcycle',
			'airplane','bus','train','truck','boat','traffic light',
			'fire hydrant','stop sign','parking meter','bench','bird',
			'cat','dog','horse','sheep','cow','elephant','bear','zebra',
			'giraffe','backpack','umbrella','handbag','tie','suitcase',
			'frisbee','skis','snowboard','sports ball','kite','baseball bat',
			'baseball glove','skateboard','surfboard','tennis racket','bottle',
			'wine glass','cup','fork','knife','spoon','bowl','banana','apple',
			'sandwich','orange','broccoli','carrot','hot dog','pizza',
			'donut','cake','chair','couch','potted plant','bed','dining table',
			'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
			'oven','toaster','sink','refrigerator','book','clock','vase','scissors',
			'teddy bear','hair drier','toothbrush')
'''

# CLASSES = ('__background__','person','bike','motorbike','car','bus')

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)
    
def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
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
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    if cfg.TEST.HAS_RPN:
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
    else:
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        if cfg.IS_MULTISCALE:
            if cfg.IS_EXTRAPOLATING:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES)
            else:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)
        else:
            blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)

    return blobs, im_scale_factors
    
    
def im_detect(sess, net, im, boxes=None):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    # forward pass
    if cfg.TEST.HAS_RPN:
        feed_dict={net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
    else:
        feed_dict={net.data: blobs['data'], net.rois: blobs['rois'], net.keep_prob: 1.0}

    cls_score, cls_prob, bbox_pred, rois = \
        sess.run([net.get_output('cls_score'), net.get_output('cls_prob'), net.get_output('bbox_pred'),net.get_output('rois')],\
                 feed_dict=feed_dict)
    
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]


    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = cls_score
    else:
        # use softmax estimated probabilities
        scores = cls_prob

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes


def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name, CONF_THRESH):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    parser.add_argument('--conf', dest='conf', help='Confidence threshold [0.8]',
                        default=0.8, type=float)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    else:
    	raise Exception('Missing config file')
        
    model_ext = glob.glob(args.model + '.*')
    if args.model == ' ' or not ( os.path.exists(args.model) or len(model_ext) == 3 ) :
        print ('current path is ' + os.path.abspath(__file__))
        raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    if len(model_ext) == 3: # new checkpoint file
    	print ('Loading model and meta {:s}...'.format(args.model+'.meta')),
    	#saver = tf.train.import_meta_graph(args.model+'.meta')
    	saver = tf.train.Saver()
    	saver.restore(sess, args.model)
    else: # old checkpoint file
    	print ('Loading network {:s}... '.format(args.demo_net)),
    	saver = tf.train.Saver()
    	saver.restore(sess, args.model)
    
    print (' done.')

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(im_name)
        demo(sess, net, im_name, args.conf)

    plt.show()
