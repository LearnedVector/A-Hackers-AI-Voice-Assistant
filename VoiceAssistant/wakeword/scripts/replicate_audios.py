import os
import argparse
import shutil

def main(args):
    ones = os.listdir(args.wakewords_dir)
    dest_dir = os.mkdir(args.wakewords_dir+'subfolder')
    os.listdir()
    for file in ones:
        if file.endswith(".wav") or file.endswith(".mp3"):
            for i in range(args.copy_number):
                # copy
                dest_dir = args.copy_destination
                srcFile = os.path.join(args.wakewords_dir, file)
                shutil.copy(srcFile, dest_dir)
                # rename the file in the subfolder
                dst_file = os.path.join(dest_dir, file)
                new_dst_file = os.path.join(dest_dir, str(i) + "_" + file)
                os.rename(dst_file, new_dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to replicate the wakeword clips by a bunch of times.

    """
    )
    parser.add_argument('--wakewords_dir', type=str, default=None, required=True,
                        help='directory of clips with wakewords')

    parser.add_argument('--copy_destination', type=str, default=None, required=True,
                        help='directory of the destinations of the wakewords copies')

    parser.add_argument('--copy_number', type=int, default=100, required=False,
                        help='the number of copies you want')

    args = parser.parse_args()

    main(args)