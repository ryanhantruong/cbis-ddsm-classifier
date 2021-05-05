import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path

(ds_train,ds_test,ds_valid),info = tfds.load('curated_breast_imaging_ddsm/patches', split=['train','test','validation'], shuffle_files=True,
              with_info=True)

train_save = ds_train.map(lambda x: (x['id'],x['image'],x['label']))
dir = 'cbis-ddsm-patches/train'

for id,im,lbl in iter(train_save):
    id = id.numpy().decode('utf-8')
    data = tf.io.encode_png(im).numpy()
    folder = info.features["label"].int2str(lbl.numpy())
    fname = '-'.join( id.split('/')) + '.png'
    fdir = '/'.join([dir,folder,fname])
    fdir = Path(fdir)
    if not fdir.parent.exists(): fdir.parent.mkdir(parents=True)
    f = fdir.open('wb')
    f.write(data)
    f.close()


test_save = ds_test.map(lambda x: (x['id'],x['image'],x['label']))
dir = 'cbis-ddsm-patches/test'

for id,im,lbl in iter(test_save):
    id = id.numpy().decode('utf-8')
    data = tf.io.encode_png(im).numpy()
    folder = info.features["label"].int2str(lbl.numpy())
    fname = '-'.join( id.split('/')) + '.png'
    fdir = '/'.join([dir,folder,fname])
    fdir = Path(fdir)
    if not fdir.parent.exists(): fdir.parent.mkdir(parents=True)
    f = fdir.open('wb')
    f.write(data)
    f.close()

 valid_save = ds_valid.map(lambda x: (x['id'],x['image'],x['label']))
 dir = 'cbis-ddsm-patches/valid'

 for id,im,lbl in iter(valid_save):
     id = id.numpy().decode('utf-8')
     data = tf.io.encode_png(im).numpy()
     folder = info.features["label"].int2str(lbl.numpy())
     fname = '-'.join( id.split('/')) + '.png'
     fdir = '/'.join([dir,folder,fname])
     fdir = Path(fdir)
     if not fdir.parent.exists(): fdir.parent.mkdir(parents=True)
     f = fdir.open('wb')
     f.write(data)
     f.close()
