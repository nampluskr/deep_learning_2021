import os
import pathlib
import numpy as np
import pickle

## http://yann.lecun.com/exdb/mnist/
server = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

url_train_images = os.path.join(server, "train-images-idx3-ubyte.gz")
url_train_labels = os.path.join(server, "train-labels-idx1-ubyte.gz")
url_test_images  = os.path.join(server, "t10k-images-idx3-ubyte.gz")
url_test_labels  = os.path.join(server, "t10k-labels-idx1-ubyte.gz")

urls = [url_train_images, url_train_labels, url_test_images, url_test_labels]
data_path  = os.path.join(os.getcwd(), 'data_fashion_mnist')


def download(url, data_path):
    import requests
    from tqdm import tqdm

    files_size = int(requests.head(url).headers["content-length"])
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(data_path, os.path.basename(url))
    chunk_size = 1024

    pbar = tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=files_size,
            desc=os.path.basename(url))
    with requests.get(url, stream=True) as req, open(file_path, 'wb') as file:
         for chunk in req.iter_content(chunk_size=chunk_size):
            data_size = file.write(chunk)
            pbar.update(data_size)
    pbar.close()


def extract(filename, image=False):
    import gzip

    with gzip.open(filename, 'rb') as f:
        if image:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            data = data.reshape(-1, 28, 28)
        else:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data


def load_data():
    if not pathlib.Path(data_path).exists():
        for url in urls:
            download(url, data_path)
    else:
        for file in os.listdir(data_path):
            print(file)

    path_train_images = os.path.join(data_path, os.path.basename(url_train_images))
    path_train_labels = os.path.join(data_path, os.path.basename(url_train_labels))
    path_test_images  = os.path.join(data_path, os.path.basename(url_test_images))
    path_test_labels  = os.path.join(data_path, os.path.basename(url_test_labels))

    train_images = extract(path_train_images, image=True)
    train_labels = extract(path_train_labels)
    test_images  = extract(path_test_images, image=True)
    test_labels  = extract(path_test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def split_data(data, split_ratio=0.8, seed=111):
    x, y = data
    cnt = int(x.shape[0] * split_ratio)

    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    data1 = x[indices[:cnt]], y[indices[:cnt]]
    data2 = x[indices[cnt:]], y[indices[cnt:]]

    return data1, data2


if __name__ == "__main__":

    ## Load data
    train_data, test_data = load_data()

    x_train, y_train = train_data
    x_test, y_test = test_data

    print("Train images:", type(x_train), x_train.shape, x_train.dtype)
    print("Train labels:", type(y_train), y_train.shape, y_train.dtype)
    print("Test  images:", type(x_test), x_test.shape, x_test.dtype)
    print("Test  labels:", type(y_test), y_test.shape, y_test.dtype)

    ## Show the 1st image
    images, labels = x_train, y_train
    print("Image shape:", images.shape)

    import matplotlib.pyplot as plt
    plt.imshow(images[0], cmap='gray_r')
    plt.title(labels[0], fontsize=15)
    plt.axis('off')
    plt.show()