from folder2lmdb import ImageFolderLMDB,ImageFolderLMDB_old
import lmdb
from torch.utils.data import DataLoader
import umsgpack
import os
path = "images/.lmdb"
dst = ImageFolderLMDB(path, None, None)
loader = DataLoader(dst, batch_size=64)
for x in loader:
	print(x)

