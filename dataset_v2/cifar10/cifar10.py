import os
import pathlib
import numpy as np
import pickle


url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
data_path = os.path.join(os.getcwd(), 'data_cifar10')
file_path = os.path.join(data_path, os.path.basename(url))
raw_path = os.path.join(data_path, "cifar-10-batches-py")


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


def extract(file_path):
    import tarfile

    with tarfile.open(file_path, 'r:gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, os.path.dirname(file_path))
    pathlib.Path(file_path).unlink()


def load_data():
    def unpickle(filename):
        with open(filename, 'rb') as f:
            file = pickle.load(f, encoding='bytes')

        x = np.array(file[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.array(file[b'labels'])
        return x, y

    if not pathlib.Path(data_path).exists():
        download(url, data_path)
        extract(file_path)

    filenames = [os.path.join(raw_path, 
            "data_batch_" + str(i+1)) for i in range(5)]

    images, labels = [], []
    for filename in filenames:
        x, y = unpickle(filename)
        images.append(x); labels.append(y)

    x_train = np.concatenate(images, axis=0)
    y_train = np.concatenate(labels, axis=0)

    filename = os.path.join(raw_path, "test_batch")
    x_test, y_test = unpickle(filename)

    return (x_train, y_train), (x_test, y_test)


def split_data(data, split_ratio=0.8, seed=111):
    x, y = data
    cnt = int(x.shape[0] * split_ratio)

    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    data1 = x[indices[:cnt]], y[indices[:cnt]]
    data2 = x[indices[cnt:]], y[indices[cnt:]]

    return data1, data2


def load_classes():
    filename = os.path.join(raw_path, "batches.meta")
    with open(filename, 'rb') as f:
        dict = pickle.load(f)
    return dict['label_names']


if __name__ == "__main__":

    ## Load data and classes
    train_data, test_data = load_data()

    x_train, y_train = train_data
    x_test, y_test = test_data

    print("Train images:", type(x_train), x_train.shape, x_train.dtype)
    print("Train labels:", type(y_train), y_train.shape, y_train.dtype)
    print("Test  images:", type(x_test), x_test.shape, x_test.dtype)
    print("Test  labels:", type(y_test), y_test.shape, y_test.dtype)

    classes = load_classes()
    print("Classes:", classes)

    ## Show the 1st image
    images, labels = x_train, y_train
    print("Image shape:", images.shape)

    import matplotlib.pyplot as plt
    plt.imshow(images[0])
    plt.title(classes[labels[0]], fontsize=15)
    plt.axis('off')
    plt.show()

