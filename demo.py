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
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

# progress-obj
CLASSES = ('__background__', # always index 0
           'tide', 'spray_bottle_a', 'waterpot', 'sugar',
           'red_bowl', 'clorox', 'shampoo', 'downy', 'salt',
           'toy', 'detergent', 'scotch_brite', 'cola',
           'blue_cup', 'ranch')

'''
# VOC
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
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

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
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
    	print ('Loading meta data {:s}...'.format(args.model+'.meta')),
    	saver = tf.train.import_meta_graph(args.model+'.meta')
    	saver.restore(sess, args.model)
    	sess.run(tf.global_variables_initializer())
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
