import os
import numpy as np
import cloud_net_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from skimage.io import imread,imsave,imshow
import numpy as np
from skimage.transform import resize
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
KTF.set_session(session)
weights_path = '/home/wxw/.local/share/cloud-net/Cloud-Net/38-Cloud dataset/Cloud-Net_trained_on_38-Cloud_training_patches.h5'
model = cloud_net_model.model_arch(input_rows=384,
                                       input_cols=384,
                                       num_of_channels=4,
                                       num_of_classes=1)
model.load_weights(weights_path)
img=imread('/home/wxw/.local/share/cloud-net/datas/Area5/GF4_PMS_E122.9_N30.0_20210213_L1A0000357629.jpg')
nir=imread('/home/wxw/.local/share/cloud-net/datas/Area5/GF4_IRS_E122.9_N30.0_20210213_L1A0000357629.tiff')
nir=nir/np.max(nir)
img=img/255
img_red=img[:,:,0]
img_green=img[:,:,1]
img_blue=img[:,:,2]
image = np.stack((img_red, img_green, img_blue, nir), axis=-1)
image = resize (image, ( 384, 384), preserve_range=True, mode='symmetric')
image=np.array([image])
result=model.predict(image)
print(result.shape)
