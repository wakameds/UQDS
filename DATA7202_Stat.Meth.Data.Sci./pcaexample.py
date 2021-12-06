######################################################################
# PCA image compression
######################################################################

from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

img = Image.open("mribrain.jpg").convert('L')

# load 2D array
im_arr = np.frombuffer(img.tobytes(), dtype=np.uint8)
im_arr = im_arr.reshape((img.size[1], img.size[0]))  

# PCA
ncomp = 80 # change the number of components
pca = PCA(n_components=ncomp)
X_pca = pca.fit_transform(im_arr)
   
X_back = pca.inverse_transform(X_pca)
X_back[X_back<1] = 0
X_back = np.floor(X_back)
X_back = X_back.astype(np.uint8)  
    
img_to_show = Image.fromarray(X_back,'L')
print("# components = ",ncomp, " variance explained = ", np.sum(pca.explained_variance_ratio_),
      " compression rate = ", 1 - (ncomp/512))
img_to_show

