# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from os.path import join
from glob import glob
import os


def get_arguments():
    parser = ArgumentParser(
        description='This script creates K different splits for files in a dataset. '
        'It expects one folder for each class')
    parser.add_argument("root_dir", help="Where the images are located")
    parser.add_argument("train_n", type=int, help="Number of training samples")
    parser.add_argument("test_n", type=int, help="Number of testing samples")
    parser.add_argument("n_splits", type=int, help="Number of splits to make")
    parser.add_argument("output_folder")
    parser.add_argument("--split_prefix", default="", help="Optional prefix for split files")
    parser.add_argument("--image_extensions", default=["jpg", "jpeg", "png"],
                        help="Extensions the script will look for")
    args = parser.parse_args()
    return args


def build_class_dataset(root_dir, image_extensions):
    class2path = {}
    dirs = os.listdir(root_dir)
    import ipdb; ipdb.set_trace()
    for dir in dirs:
        images = [glob(join(root_dir, dir) + '*' + ext) for ext in image_extensions]
        class2path[dir] = images
    return class2path


def create_split(class2path, n_train, n_test, outpath):
    return None


if __name__ == '__main__':
    args = get_arguments()
    class2path = build_class_dataset(args.root_dir, args.image_extensions)
    for i in range(args.n_splits):
        outpath = join(args.output_folder, args.split_prefix)
        create_split(class2path, args.train_n, args.test_n, outpath)
