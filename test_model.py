
from tensorflow.keras.models import load_model
from skimage.io import imread, imshow
from skimage.transform import resize
import numpy as np

model=load_model("unets_model.h5")


def do_pred(img):

      img_width=128
      img_height=128
      img_channels=3
     
      X_test=np.zeros((1,img_height,img_width,img_channels),dtype=np.uint8)

      img=imread(img)
      img=resize(img,(img_height,img_width,img_channels),mode="constant",preserve_range=True)
      X_test[0]=img
     
      pred_mask=model.predict(X_test)
      pred_mask=pred_mask.reshape(128,128)
      return pred_mask
  
