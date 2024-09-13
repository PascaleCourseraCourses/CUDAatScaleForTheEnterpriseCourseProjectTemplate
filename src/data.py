import numpy as np
from PIL import Image
import os
import struct
import csv

def read_idx_images(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        # Read image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    
    return images

def read_idx_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack(">II", f.read(8))

        # Read label data
        labels = np.fromfile(f, dtype=np.uint8)
    
    return labels

def save_images(images, img_dir):
    os.makedirs(img_dir, exist_ok=True)
    for i in range(images.shape[0]):
        img = Image.fromarray(images[i], 'L')
        img.save(os.path.join(img_dir, '{}.png'.format(i)))
    
def save_labels(labels, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for label in labels:
            writer.writerow([label])

def main():
    # Paths to MNIST files
    train_images_file = './data/train-images.idx3-ubyte'
    train_labels_file = './data/train-labels.idx1-ubyte'
    test_images_file = './data/t10k-images.idx3-ubyte'
    test_labels_file = './data/t10k-labels.idx1-ubyte'

    # Read data
    train_images = read_idx_images(train_images_file)
    train_labels = read_idx_labels(train_labels_file)
    test_images = read_idx_images(test_images_file)
    test_labels = read_idx_labels(test_labels_file)

    # Save images
    save_images(train_images, './data/train/mnist_images')
    save_images(test_images, './data/test/mnist_images')

    # Save labels
    save_labels(train_labels, './data/train/mnist_labels.csv')
    save_labels(test_labels, './data/test/mnist_labels.csv')

if __name__ == '__main__':
    main()
