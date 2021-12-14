# Dataset

## User-defined Functions
---

### 데이터 파일 다운로드
```python
def download_data(url, data_path):
    import requests
    from tqdm import tqdm

    files_size = int(requests.head(url).headers["content-length"])
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(data_path, os.path.basename(url))

    pbar = tqdm(total=files_size, unit='B', unit_scale=True, unit_divisor=1024,
                ascii=True, desc=os.path.basename(url))

    with requests.get(url, stream=True) as req, open(file_path, 'wb') as file:
        for chunk in req.iter_content(chunk_size=1024):
            data_size = file.write(chunk)
            pbar.update(data_size)
        pbar.close()
```
### 압축파일 풀기
```python
def extract(file_path, extract_path):
    import shutil

    print(">> Extracting", os.path.basename(file_path))
    shutil.unpack_archive(file_path, extract_path)
    pathlib.Path(file_path).unlink()
    print(">> Completed!")
```

### gzip 파일 불러오기 (MNIST, Fashion MNIST, Extened MNIST)
```python
def load(file_path, image=False):
    import gzip
    
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if image else 8)
    return data.reshape(-1, 28, 28) if image else data
```

## MNIST Dataset
---
### 데이터셋 불러오기
```python
def load_mnist(data_path, download=False):
    """ The MNIST dataset: http://yann.lecun.com/exdb/mnist/ """

    url = "http://yann.lecun.com/exdb/mnist/"

    train_images = "train-images-idx3-ubyte.gz"
    train_labels = "train-labels-idx1-ubyte.gz"
    test_images = "t10k-images-idx3-ubyte.gz"
    test_labels = "t10k-labels-idx1-ubyte.gz"
    filenames = [train_images, train_labels, test_images, test_labels]

    data_path = os.path.abspath(data_path)
    if not pathlib.Path(data_path).exists() and download:
        for filename in filenames:
            download_data(os.path.join(url, filename), data_path)

    x_train = load(os.path.join(data_path, train_images), image=True)
    y_train = load(os.path.join(data_path, train_labels), image=False)
    x_test = load(os.path.join(data_path, test_images), image=True)
    y_test = load(os.path.join(data_path, test_labels), image=False)

    return (x_train, y_train), (x_test, y_test)
```

### 클래스 이름 불러오기
```python
def load_class_names():
    return [str(i) for i in range(10)]
```

### 사용예
```python
data_path = "./data/data_mnist"
print(os.path.abspath(data_path))

train_data, test_data = load_fashion_mnist(data_path, download=True)
images, labels = train_data

print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())

class_names = load_class_names()
print(class_names)
```

## Fashion MNIST Dataset
---

### 데이터셋 불러오기
```python
def load_fashion_mnist(data_path, download=False):
    """ https://github.com/zalandoresearch/fashion-mnist """

    url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

    train_images = "train-images-idx3-ubyte.gz"
    train_labels = "train-labels-idx1-ubyte.gz"
    test_images = "t10k-images-idx3-ubyte.gz"
    test_labels = "t10k-labels-idx1-ubyte.gz"
    filenames = [train_images, train_labels, test_images, test_labels]

    data_path = os.path.abspath(data_path)
    if not pathlib.Path(data_path).exists() and download:
        for filename in filenames:
            download_data(os.path.join(url, filename), data_path)

    x_train = load(os.path.join(data_path, train_images), image=True)
    y_train = load(os.path.join(data_path, train_labels), image=False)
    x_test = load(os.path.join(data_path, test_images), image=True)
    y_test = load(os.path.join(data_path, test_labels), image=False)

    return (x_train, y_train), (x_test, y_test)
```
### 클래스 이름 불러오기
```python
def load_class_names():
    return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

### 사용예
```python
data_path = "./data/data_fashion_mnist"
print(os.path.abspath(data_path))

train_data, test_data = load_fashion_mnist(data_path, download=True)
images, labels = train_data

print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())

class_names = load_class_names()
print(class_names)
```


## Extened MNIST
---

### 파일 이름 딕셔너리
```python
emnist = dict(
    balanced = {
        'train_images':'emnist-balanced-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-balanced-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-balanced-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-balanced-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-balanced-mapping.txt',},
    byclass = {
        'train_images':'emnist-byclass-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-byclass-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-byclass-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-byclass-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-byclass-mapping.txt',},
    bymerge = {
        'train_images':'emnist-bymerge-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-bymerge-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-bymerge-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-bymerge-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-bymerge-mapping.txt',},
    digits = {
        'train_images':'emnist-digits-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-digits-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-digits-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-digits-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-digits-mapping.txt',},
    letters = {
        'train_images':'emnist-letters-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-letters-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-letters-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-letters-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-letters-mapping.txt',},
    mnist = {
        'train_images':'emnist-mnist-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-mnist-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-mnist-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-mnist-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-mnist-mapping.txt',})
```
### 데이터셋 불러오기
```python
def load_extended_mnist(data_path, split_name='mnist', download=False):
    """ https://www.nist.gov/itl/products-and-services/emnist-dataset """

    assert split_name in emnist.keys()

    url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/"
    data_path = os.path.abspath(data_path)
    
    if not pathlib.Path(data_path).exists() and download:
        download_data(os.path.join(url, "gzip.zip"), data_path)
        extract(os.path.join(data_path, "gzip.zip"), data_path)

    train_images = emnist[split_name]['train_images']
    train_labels = emnist[split_name]['train_labels']
    test_images = emnist[split_name]['test_images']
    test_labels = emnist[split_name]['test_labels']

    x_train = load(os.path.join(data_path, 'gzip', train_images), image=True)
    y_train = load(os.path.join(data_path, 'gzip', train_labels), image=False)
    x_test = load(os.path.join(data_path, 'gzip', test_images), image=True)
    y_test = load(os.path.join(data_path, 'gzip', test_labels), image=False)

    return (x_train, y_train), (x_test, y_test)
```

### 클래스 이름 불러오기
```python
def load_class_names(data_path, split_name='mnist'):
    assert split_name in emnist.keys()

    data_path = os.path.abspath(data_path)
    path = os.path.join(data_path, 'gzip', emnist[split_name]['mapping'])
    class_names = [chr(int(cls)) for cls in np.genfromtxt(path)[:, 1]]
    return class_names
```

### 사용예
```python
data_path = "./data_extened_mnist"
print(os.path.abspath(data_path))

train_data, test_data = load_extended_mnist(data_path, split_name='digits', download=True)
train_images, train_labels = train_data
print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())

class_names = load_class_names(data_path, split_name='digits')
print(class_names)
```