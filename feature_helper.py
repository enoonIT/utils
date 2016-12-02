# -*- coding: utf-8 -*-
import pickle
from argparse import ArgumentParser
from os.path import join
import feature_handler_v3
import h5py
import os


def get_arguments():
    parser = ArgumentParser(
        description='This tool extracts features and saves them into a pickle dictionary')
    parser.add_argument("data_dir", help="Where the images are located")
    parser.add_argument("filelist", help="File containing the list of all files")
    parser.add_argument("net_proto", help="Net deploy prototxt")
    parser.add_argument("net_model", help="Net model")
    parser.add_argument("output_filename")
    parser.add_argument("--mean_pixel", type=float)
    parser.add_argument("--mean_file")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--layer_name", help="Default is FC7", default='fc7')
    parser.add_argument("--use_cpu", action="store_true", help="If set false, will force CPU inference")
    parser.add_argument("--center_data", action="store_true", help="If set will center the data")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--opt_ext", default="", help="Extension to add to the filenames")
    parser.add_argument("--patchMode", action="store_true")
    parser.add_argument("--data_collection", action="store_true")
    args = parser.parse_args()
    return args


def handle_patches(all_files, f_extractor, directory, rootPath):
    #    import ipdb; ipdb.set_trace()
    if not os.path.exists(directory):
        os.makedirs(directory)
    mode = f_extractor.get_mode()
    (height, width) = f_extractor.get_input_size()
    c = 0
    for file in all_files:
        # import ipdb; ipdb.set_trace()
        (feats, meta) = f_extractor.get_grid_features(join(rootPath, file.strip()), mode, height, width)
        f = h5py.File(os.path.join(directory, os.path.basename(file.strip()) + "_" + str(c) + ".hdf5"), "w")
        c += 1
        data = f.create_dataset("feats", feats.shape, compression="gzip", compression_opts=9, dtype="float32")
        pos = f.create_dataset("position", (meta.shape[0], 2), compression="gzip", compression_opts=9, dtype="float32")
        level = f.create_dataset("level", (meta.shape[0], ), compression="gzip", compression_opts=9, dtype="uint16")
        depth = f.create_dataset("depth", (meta.shape[0], ), compression="gzip", compression_opts=9, dtype="float32")
        f.attrs['relative_path'] = file.strip()
        data[:] = feats
        pos[:] = meta[:, 0:2]
        level[:] = meta[:, 2]
        depth[:] = meta[:, 3]
        f.close()


def make_features(args):
    print "Starting feature generation procedure..."
    f_extractor = feature_handler_v3.FeatureCreator(
        args.net_proto, args.net_model, args.mean_pixel, args.mean_file, not args.use_cpu,
        layer_name=args.layer_name, gpu_id=args.gpu_id)
    f_extractor.batch_size = args.batch_size
    f_extractor.center_data = args.center_data
    f_extractor.set_data_scale(args.scale)
    f_extractor.data_prefix = args.data_dir
    all_lines = open(args.filelist, 'rt').readlines()
    if args.patchMode:
        handle_patches(all_lines, f_extractor, args.output_filename, args.data_dir)
    all_lines = [join(args.data_dir, line.strip() + args.opt_ext) for line in all_lines]
    if args.data_collection:
        stats = f_extractor.prepare_features_iter(all_lines)
        with open(args.output_filename, 'wb') as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
    else:
        # preload all features so that they are handled in batches
        f_extractor.prepare_features(all_lines)
        with open(args.output_filename, 'wb') as f:
            pickle.dump(f_extractor.features, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = get_arguments()
    make_features(args)
