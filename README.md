# Image2LMDB
Convert image folder to lmdb, adapted from https://github.com/Lyken17/Efficient-PyTorch
```
.
├── folder2lmdb.py
├── img
│   ├── train
│   │   ├── bar_dir
│   │   │   ├── 100000.jpg
│   │   │   ├── 100001.jpg
│   │   │   ├── 100002.jpg
│   │   │   ├── 100003.jpg
│   │   │   ├── 100004.jpg
│   │   │   ├── 100005.jpg
│   │   │   ├── 100006.jpg
│   │   │   ├── 100007.jpg
│   │   │   ├── 100008.jpg
│   │   │   └── 100009.jpg
│   │   └── foo_dir
│   │       ├── 100000.jpg
│   │       ├── 100001.jpg
│   │       ├── 100002.jpg
│   │       ├── 100003.jpg
│   │       ├── 100004.jpg
│   │       ├── 100005.jpg
│   │       ├── 100006.jpg
│   │       ├── 100007.jpg
│   │       ├── 100008.jpg
│   │       └── 100009.jpg
│   ├── train_images_idx.txt
│   ├── train.lmdb
│   └── train.lmdb-lock
├── main.py
├── __pycache__
│   └── folder2lmdb.cpython-36.pyc
├── README.md
└── requirements.txt
```

## Convert image folder to lmdb
```python
python folder2lmdb.py img
````

## Test it
```python
python main.py img/train.lmdb
```


```
key 0
key 1
torch.Size([2, 224, 224, 3])
key 2
key 3
torch.Size([2, 224, 224, 3])
key 4
key 5
torch.Size([2, 224, 224, 3])
key 6
key 7
torch.Size([2, 224, 224, 3])
key 8
key 9
torch.Size([2, 224, 224, 3])
key 10
key 11
torch.Size([2, 224, 224, 3])
key 12
key 13
torch.Size([2, 224, 224, 3])
key 14
key 15
torch.Size([2, 224, 224, 3])
key 16
key 17
torch.Size([2, 224, 224, 3])
key 18
key 19
torch.Size([2, 224, 224, 3])
```



## Original Repo:
https://github.com/Lyken17/Efficient-PyTorch
