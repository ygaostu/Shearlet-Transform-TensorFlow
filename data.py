import tensorflow as tf
from glob import glob
import os
import logging
import shutil

def rm_dir(dname):
    if os.path.exists(dname):
        shutil.rmtree(dname)

def rm_mk_dir(dname):
    rm_dir(dname)
    os.makedirs(dname)

def create_image_dataset(train=True, params={}):
    if train:
        epi_folders = [] 
    else:
        epi_folders = sorted([name for name in glob("{}/*".format(params["validate_path"])) if os.path.isdir(name)])

    save_path = params["save_path"]
    rm_mk_dir(save_path)

    filenames_epi = []
    filenames_mask = []
    filenames_to_save = []

    for epi_folder in epi_folders:
        im_names_epi = sorted(glob(os.path.join(epi_folder, "*rgb.png")))
        im_names_mask = [name.replace("rgb.png", "mask.png") for name in im_names_epi]

        epi_folder_basename = os.path.basename(epi_folder)
        os.makedirs(os.path.join(save_path, epi_folder_basename), exist_ok=True)
        im_names_to_save = [os.path.join(save_path, epi_folder_basename, os.path.basename(n)) for n in im_names_epi]

        filenames_epi += im_names_epi
        filenames_mask += im_names_mask
        filenames_to_save += im_names_to_save

    filenames_epi = tf.constant(filenames_epi)
    filenames_mask = tf.constant(filenames_mask)
    filenames_to_save = tf.constant(filenames_to_save)

    def _parse_function(name_train, name_mask, name_save):
        im_epi = tf.image.decode_png(tf.read_file(name_train))
        im_mask = tf.image.decode_png(tf.read_file(name_mask))

        im_epi = tf.cast(im_epi, tf.float32)
        im_mask = tf.cast(im_mask, tf.float32)/255.

        im_epi = tf.transpose(im_epi, perm = [2, 0, 1])
        im_mask = tf.transpose(im_mask, perm = [2, 0, 1])
        im_mask = tf.tile(im_mask, [3, 1, 1])
        
        features = {
            "im_epi": im_epi, 
            "im_mask": im_mask,
            "name_save": name_save,
        }

        return features, tf.constant(0)

    d = tf.data.Dataset.from_tensor_slices((filenames_epi, filenames_mask, filenames_to_save))
    d = d.map(_parse_function, num_parallel_calls=2)

    return d
