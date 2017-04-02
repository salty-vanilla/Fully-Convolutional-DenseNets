import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def plot(dst_prefix, src_path):
    dst_dir = os.path.dirname(dst_prefix)
    if dst_dir is not '':
        os.makedirs(dst_dir, exist_ok=True)

    data = pd.read_csv(src_path)
    loss = data['loss'].as_matrix()

    if 'acc' in data.columns:
        acc = data['acc'].as_matrix()

    if 'val_loss' in data.columns:
        val_loss = data['val_loss'].as_matrix()

    if 'val_acc' in data.columns:
        val_acc = data['val_acc'].as_matrix()

    plt.figure(1)
    x = np.arange(len(loss))
    plt.plot(x, loss, label="loss")
    if 'val_loss' in data.columns:
        plt.plot(x, val_loss, label="val_loss")
    plt.xlim(0.0, len(x))
    plt.legend()
    plt.savefig(dst_prefix + "_loss.png")
    plt.clf()

    if 'acc' in data.columns:
        plt.figure(2)
        plt.plot(x, acc, label="acc")
        if 'val_acc' in data.columns:
            plt.plot(x, val_acc, label="val_acc")
        plt.xlim(0.0, len(x))
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.savefig(dst_prefix + "_acc.png")
    plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dst_dir', default="images/")
    parser.add_argument('file_paths', nargs='+')
    args = parser.parse_args()

    dst_dir = args.dst_dir
    file_paths = args.file_paths

    for file_path in file_paths:
        name, ext = os.path.splitext(os.path.basename(file_path))
        dst_prefix = os.path.join(dst_dir, name)
        plot(dst_prefix, file_path)


if __name__ == "__main__":
    main()
