from folder2lmdb import ImageFolderLMDB,ImageFolderLMDB_old
import lmdb
from torch.utils.data import DataLoader
import umsgpack
import os
import fire
from torchvision.transforms import Resize

def main(path):
# path = "images/.lmdb"
    transform = Resize((224,224))
    dst = ImageFolderLMDB(path, transform, None)
    loader = DataLoader(dst, batch_size=2)
    for x in loader:
            print(x)


if __name__ == '__main__':
    fire.Fire(main)
