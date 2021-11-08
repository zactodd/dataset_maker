from dataset_maker.annotations.instance_segmentation import convert_annotation_format
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_dir", type=str, help="THe directory where the annotation images are stored.")
parser.add_argument("annotations_dir", type=str, help="The directory where the annotation images are stored.")
parser.add_argument("download_dir", type=str, help="The directory where the new annotation for is created.")
parser.add_argument("in_format", type=str, help="The format of the current annotation file(s).")
parser.add_argument("out_format", type=str, help="The format of the new annotation file(s).")


def main():
    args = parser.parse_args(sys.argv[1:])
    convert_annotation_format(**vars(args))


if __name__ == '__main__':
    main()
