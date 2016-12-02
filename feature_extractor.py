import os
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc
import time
from tqdm import tqdm
from Estimator import Estimator
os.environ['GLOG_minloglevel'] = '2'


def get_net(caffemodel, deploy_file, use_gpu=True, GPU_ID=0):
    """
    Returns an instance of caffe.Net
    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        print "Using GPU %d" % GPU_ID
        caffe.set_device(GPU_ID)
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)


def get_transformer(deploy_file, mean_file=None, mean_pixel=None):
    """
    Returns an instance of caffe.io.Transformer
    Arguments:
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(inputs={'data': dims})
    # transpose to (channels, height, width)
    t.set_transpose('data', (2, 0, 1))

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2, 1, 0))

    if mean_file:
        # set mean pixel
        print "Using mean file"
        with open(mean_file, 'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(
                    blob_dims
                ) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError(
                    'blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            print pixel
            t.set_mean('data', pixel)
    if mean_pixel:
        print "Using mean pixel %f" % mean_pixel
        t.set_mean('data', np.ones(dims[1]) * mean_pixel)

    return t


def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk
    Returns an np.ndarray (channels x width x height)
    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension
    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    try:
        image = image.resize((width, height), PIL.Image.BILINEAR)
    except:
        print "Can't load image %s" % path
        return None
    return convert_image(image, mode)


def convert_image(image, mode):
    if image.mode == 'I':
        tmp = np.array(image)
        if mode == 'RGB':
            w, h = tmp.shape
            data = np.empty((w, h, 3), dtype=np.float32)
            data[:, :, 2] = data[:, :, 1] = data[:, :, 0] = tmp
        else:
            data = tmp
    else:
        image = image.convert(mode)
        data = np.array(image)
    return data


def forward_pass(images, net, transformer, batch_size, layer_names):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)
    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer
    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    batch_size = min(batch_size, len(images))
    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:, :, np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]
    feat_holder = {}
    for layer_name in layer_names:
        fsize = net.blobs[layer_name].data.size / \
                net.blobs[layer_name].data.shape[0]
        feat_holder[layer_name] = np.empty((len(images), fsize), dtype='float32')
    todoChunks = [caffe_images[x:x + batch_size]
                  for x in xrange(0, len(caffe_images), batch_size)]
    start = time.clock()
    idx = 0
    for k, chunk in tqdm(list(enumerate(todoChunks))):
        bsize = len(chunk)
        new_shape = (bsize, ) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        net.forward()
        for layer_name in layer_names:
            features = feat_holder[layer_name]
            features[idx:idx + bsize] = net.blobs[layer_name].data.reshape(
                -1, net.blobs[layer_name].data.size / bsize).copy()
        idx += bsize
    print "It took %f" % (time.clock() - start)
    return feat_holder


class FeatureCreator:
    """This class keeps computed features in memory
    and returns them when requested"""

    def __init__(self,
                 net_proto,
                 net_weights,
                 mean_pixel=None,
                 mean_file=None,
                 use_gpu=True,
                 layer_name='fc7',
                 verbose=False,
                 gpu_id=0):
        self.net = get_net(net_weights, net_proto, use_gpu, gpu_id)
        self.transformer = get_transformer(
            net_proto, mean_pixel=mean_pixel, mean_file=mean_file)
        self.features = {}
        self.layer_name = layer_name
        self.f_size = self.net.blobs[self.layer_name].data.shape[1]
        self.batch_size = 256
        self.scale = 1
        self.verbose = verbose
        self.data_prefix = ''

    def get_mode(self):
        old_batch_size, channels, height, width = self.transformer.inputs[
            'data']
        if channels == 3:
            mode = 'RGB'
        elif channels == 1:
            mode = 'L'
        else:
            raise ValueError('Invalid number for channels: %s' % channels)
        return mode

    def get_input_size(self):
        old_batch_size, channels, height, width = self.transformer.inputs[
            'data']
        return (height, width)

    def prepare_features(self, image_files):
        old_batch_size, channels, height, width = self.transformer.inputs['data']
        mode = self.get_mode()
        print "Loading images"
        images = []
        es = Estimator()
        for image_file in image_files:
            im = load_image(image_file, height, width, mode)
            images.append(im)
            es.push(im.mean(axis=tuple((0, 1))))
        mean = self.scale * es.get_mean()
        print "Image mean: %s" % str(mean)
        if self.center_data:
            self.transformer.set_mean('data', mean)
            print "Will center data"
        # Classify the image
        print "Extracting features"
        layers = ['fc7']
        feats = forward_pass(images, self.net, self.transformer, self.batch_size, layers)
        i = 0
        for f in image_files:
            # saves only the relative path
            short_name = f.replace(self.data_prefix, '')
            if short_name[0] == '/':
                short_name = short_name[1:]
            self.features[short_name] = feats.values()[0][i].ravel() # TODO join different features
            i += 1
        self.net = None  # free video memory

    def prepare_features_stats(self, image_files, _batch_size=1024):
        old_batch_size, channels, height, width = self.transformer.inputs[
            'data']
        mode = self.get_mode()
        print "Loading images"
        images = []
        es = Estimator()
        for image_file in image_files:
            im = load_image(image_file, height, width, mode)
            images.append(im)
            es.push(im.mean(axis=tuple((0, 1))))
        mean = self.scale * es.get_mean()
        print "Image mean: %s" % str(mean)
        if self.center_data:
            self.transformer.set_mean('data', mean)
            print "Will center data"
        # Classify the image
        print "Extracting features"
        layers = ['fc6', 'fc7']
        feat_holder = forward_pass(images, self.net, self.transformer, self.batch_size, layers)

    def get_stats(layers, feat_holder):
        stats = {}
        for layer in layers:
            feats = feat_holder[layer]
            est = Estimator()
            for i in range(feats.shape[0]):
                est.push(feats[i].ravel())
            stats[layer] = (est.get_mean(), est.get_std())
        return stats

    def get_grid_features(self, image_path, mode, desH, desW):
        data = convert_image(PIL.Image.open(image_path), mode)
        (h, w) = data.shape[0:2]  # the third dim might be present for RGB
        cS = 4  # center size for depth mean
        maxSize = max((w, h))
        pSizes = (np.array([0.16, 0.32, 0.64]) * maxSize).astype('uint16')
        nLevels = pSizes.size
        # better fixed step size or variable step size?
        stepSizes = np.ones(
            nLevels, dtype='int8') * pSizes[
                0] / 2  # pSizes / 2 step size is half of feature size
        patches = [data]  # start with the whole image
        metaInfo = [[
            0.5, 0.5, 0, data[h / 2 - cS:h / 2 + cS, w / 2 - cS:w / 2 + cS]
            .mean()
        ]]
        for l in range(nLevels):
            pSize = pSizes[l]
            if pSize <= 4 or stepSizes[l] < 1:
                continue
            for x in range(0, w - pSize, stepSizes[l]):
                for y in range(0, h - pSize, stepSizes[l]):
                    crop = data[y:y + pSize, x:x + pSize]
                    patches.append(crop)
                    # x, y, patch level, depth of center area
                    metaInfo.append([(x + pSize / 2.0) / w,
                                     (y + pSize / 2.0) / h, nLevels - l,
                                     crop[pSize / 2 - cS:pSize / 2 + cS, pSize
                                          / 2 - cS:pSize / 2 + cS].mean()])
            print "%d patches, %dx%d - %d" % (len(patches), w, h, pSize)
        patches = [
            scipy.misc.imresize(patch, (desH, desW), 'bilinear')
            for patch in patches
        ]
        features = forward_pass(
            patches,
            self.net,
            self.transformer,
            batch_size=self.batch_size,
            layer_name=self.layer_name)
        return (features, np.vstack(metaInfo))

    def get_features(self, image_path):
        feats = self.features.get(image_path, None)
        if feats is None:
            print "!!! Missing features for " + image_path
        return feats

    def set_data_scale(self, scale):
        if scale is None:
            return
        self.transformer.set_raw_scale('data', scale)
        print "Set transformer raw data scale to %f" % scale
        self.scale = scale

    def do_forward_pass(self, data):
        return forward_pass(
            data,
            self.net,
            self.transformer,
            self.verbose,
            batch_size=self.batch_size,
            layer_name=self.layer_name)
