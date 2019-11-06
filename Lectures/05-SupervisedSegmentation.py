#!/usr/bin/env python
# coding: utf-8

# # ETHZ: 227-0966-00L
# # Quantitative Big Imaging
# 
# # March 21, 2019
# 
# ## Supervised Approaches

# # Reading Material
# 
# - [Introduction to Machine Learning: ETH Course](https://las.inf.ethz.ch/teaching/introml-s18)
# - [Decision Forests for Computer Vision and Medical Image Analysis](https://www.amazon.com/Decision-Computer-Analysis-Advances-Recognition/dp/1447149289/ref=sr_1_1?s=books&ie=UTF8&qid=1521704598&sr=1-1&refinements=p_27%3AAntonio+Criminisi&dpID=41fMCWUOh%252BL&preST=_SY291_BO1,204,203,200_QL40_&dpSrc=srch)
# - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
# - [U-Net Website](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

# # Overview
# 1. Methods
# 1. Pipelines
# 2. Classification
# 3. Regression
# 4. Segmentation

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})


# # Basic Methods Overview
# There are a number of methods we can use for classification, regression and both. For the simplification of the material we will not make a massive distinction between classification and regression but there are many situations where this is not appropriate. Here we cover a few basic methods, since these are important to understand as a starting point for solving difficult problems. The list is not complete and importantly Support Vector Machines are completely missing which can be a very useful tool in supervised analysis.
# A core idea to supervised models is they have a training phase and a predicting phase. 
# ## Training
# 
# The training phase is when the parameters of the model are *learned* and involve putting inputs into the model and updating the parameters so they better match the outputs. This is a sort-of curve fitting (with linear regression it is exactly curve fitting).
# 
# ## Predicting
# 
# The predicting phase is once the parameters have been set applying the model to new datasets. At this point the parameters are no longer adjusted or updated and the model is frozen. Generally it is not possible to tweak a model any more using new data but some approaches (most notably neural networks) are able to handle this. 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')

blob_data, blob_labels = make_blobs(n_samples=100,
                                    random_state=2018)
test_pts = pd.DataFrame(blob_data, columns=['x', 'y'])
test_pts['group_id'] = blob_labels
plt.scatter(test_pts.x, test_pts.y,
            c=test_pts.group_id,
            cmap='viridis')
test_pts.sample(5)


# ## Nearest Neighbor (or K Nearest Neighbors)
# The technique is as basic as it sounds, it basically finds the nearest point to what you have put in. 

# In[3]:


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
k_class = KNeighborsClassifier(1)
k_class.fit(X=np.reshape([0, 1, 2, 3], (-1, 1)),
            y=['I', 'am', 'a', 'dog'])


# In[4]:


print(k_class.predict(np.reshape([0, 1, 2, 3],
                                 (-1, 1))))


# In[5]:


print(k_class.predict(np.reshape([1.5], (1, 1))))
print(k_class.predict(np.reshape([100], (1, 1))))


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')

blob_data, blob_labels = make_blobs(n_samples=100,
                                    cluster_std=2.0,
                                    random_state=2018)
test_pts = pd.DataFrame(blob_data, columns=['x', 'y'])
test_pts['group_id'] = blob_labels
plt.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')
test_pts.sample(5)


# In[7]:


k_class = KNeighborsClassifier(1)
k_class.fit(test_pts[['x', 'y']], test_pts['group_id']) 


# In[8]:


xx, yy = np.meshgrid(np.linspace(test_pts.x.min(), test_pts.x.max(), 30),
                     np.linspace(test_pts.y.min(), test_pts.y.max(), 30),
                     indexing='ij'
                     )
grid_pts = pd.DataFrame(dict(x=xx.ravel(), y=yy.ravel()))
grid_pts['predicted_id'] = k_class.predict(grid_pts[['x', 'y']])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')
ax1.set_title('Training Data')
ax2.scatter(grid_pts.x, grid_pts.y, c=grid_pts.predicted_id, cmap='viridis')
ax2.set_title('Testing Points')


# # Stabilizing Results
# We can see here that the result is thrown off by single points, we can improve by using more than the nearest neighbor and include the average of the nearest 2 neighbors

# In[9]:


k_class = KNeighborsClassifier(4)
k_class.fit(test_pts[['x', 'y']], test_pts['group_id'])
xx, yy = np.meshgrid(np.linspace(test_pts.x.min(), test_pts.x.max(), 30),
                     np.linspace(test_pts.y.min(), test_pts.y.max(), 30),
                     indexing='ij'
                     )
grid_pts = pd.DataFrame(dict(x=xx.ravel(), y=yy.ravel()))
grid_pts['predicted_id'] = k_class.predict(grid_pts[['x', 'y']])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')
ax1.set_title('Training Data')
ax2.scatter(grid_pts.x, grid_pts.y, c=grid_pts.predicted_id, cmap='viridis')
ax2.set_title('Testing Points')


# ## Linear Regression
# Linear regression is a fancy-name for linear curve fitting, fitting a line through points (sometimes in more than one dimension). It is a very basic method, but is easy to understand, interpret and fast to compute

# In[10]:


from sklearn.linear_model import LinearRegression
import numpy as np
l_reg = LinearRegression()
l_reg.fit(X=np.reshape([0, 1, 2, 3], (-1, 1)),
          y=[10, 20, 30, 40])
l_reg.coef_, l_reg.intercept_


# In[11]:


print(l_reg.predict(np.reshape([0, 1, 2, 3], (-1, 1))))
print(-100, '->', l_reg.predict(np.reshape([-100], (1, 1))))
print(500, '->', l_reg.predict(np.reshape([500], (1, 1))))


# In[12]:


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
blob_data, blob_labels = make_blobs(centers=2, n_samples=100,
                                    random_state=2018)
test_pts = pd.DataFrame(blob_data, columns=['x', 'y'])
test_pts['group_id'] = blob_labels
plt.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')
test_pts.sample(5)


# In[13]:


l_reg = LinearRegression()
l_reg.fit(test_pts[['x', 'y']], test_pts['group_id'])
print('Slope', l_reg.coef_)
print('Offset', l_reg.intercept_)


# In[14]:


xx, yy = np.meshgrid(np.linspace(test_pts.x.min(), test_pts.x.max(), 20),
                     np.linspace(test_pts.y.min(), test_pts.y.max(), 20),
                     indexing='ij'
                     )
grid_pts = pd.DataFrame(dict(x=xx.ravel(), y=yy.ravel()))
grid_pts['predicted_id'] = l_reg.predict(grid_pts[['x', 'y']])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')
ax1.set_title('Training Data')
ax2.scatter(grid_pts.x, grid_pts.y, c=grid_pts.predicted_id, cmap='viridis')
ax2.set_title('Testing Points')
ax3.imshow(grid_pts.predicted_id.values.reshape(
    xx.shape).T[::-1], cmap='viridis')
ax3.set_title('Test Image')


# ## Trees

# In[15]:


from sklearn.tree import export_graphviz
import graphviz


def show_tree(in_tree):
    return graphviz.Source(export_graphviz(in_tree, out_file=None))


# In[16]:


from sklearn.tree import DecisionTreeClassifier
import numpy as np
d_tree = DecisionTreeClassifier()
d_tree.fit(X=np.reshape([0, 1, 2, 3], (-1, 1)),
           y=[0, 1, 0, 1])


# In[17]:


show_tree(d_tree)


# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')

blob_data, blob_labels = make_blobs(n_samples=100,
                                    random_state=2018)
test_pts = pd.DataFrame(blob_data, columns=['x', 'y'])
test_pts['group_id'] = blob_labels
plt.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')
test_pts.sample(5)


# In[19]:


d_tree = DecisionTreeClassifier()
d_tree.fit(test_pts[['x', 'y']],
           test_pts['group_id'])
show_tree(d_tree)


# In[20]:


xx, yy = np.meshgrid(np.linspace(test_pts.x.min(), test_pts.x.max(), 20),
                     np.linspace(test_pts.y.min(), test_pts.y.max(), 20),
                     indexing='ij'
                     )
grid_pts = pd.DataFrame(dict(x=xx.ravel(), y=yy.ravel()))
grid_pts['predicted_id'] = d_tree.predict(grid_pts[['x', 'y']])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')
ax1.set_title('Training Data')
ax2.scatter(grid_pts.x, grid_pts.y, c=grid_pts.predicted_id, cmap='viridis')
ax2.set_title('Testing Points')


# ## Forests
# Forests are basically the idea of taking a number of trees and bringing them together. So rather than taking a single tree to do the classification, you divide the samples and the features to make different trees and then combine the results. One of the more successful approaches is called [Random Forests](https://en.wikipedia.org/wiki/Random_forest) or as a [video](https://www.youtube.com/watch?v=loNcrMjYh64)

# In[84]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')

blob_data, blob_labels = make_blobs(n_samples=1000,
                                    cluster_std=3,
                                    random_state=2018)
test_pts = pd.DataFrame(blob_data, columns=['x', 'y'])
test_pts['group_id'] = blob_labels
plt.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')


# In[85]:


from sklearn.ensemble import RandomForestClassifier
rf_class = RandomForestClassifier(n_estimators=5, random_state=2018)
rf_class.fit(test_pts[['x', 'y']],
             test_pts['group_id'])
print('Build ', len(rf_class.estimators_), 'decision trees')


# In[86]:


show_tree(rf_class.estimators_[0])


# In[24]:


show_tree(rf_class.estimators_[1])


# In[87]:


xx, yy = np.meshgrid(np.linspace(test_pts.x.min(), test_pts.x.max(), 20),
                     np.linspace(test_pts.y.min(), test_pts.y.max(), 20),
                     indexing='ij'
                     )
grid_pts = pd.DataFrame(dict(x=xx.ravel(), y=yy.ravel()))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 3), dpi=150)
ax1.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')
ax1.set_title('Training Data')
ax2.scatter(grid_pts.x, grid_pts.y, c=rf_class.predict(
    grid_pts[['x', 'y']]), cmap='viridis')
ax2.set_title('Random Forest Classifier')

ax3.scatter(grid_pts.x, grid_pts.y, c=rf_class.estimators_[
            0].predict(grid_pts[['x', 'y']]), cmap='viridis')
ax3.set_title('First Decision Tree')

ax4.scatter(grid_pts.x, grid_pts.y, c=rf_class.estimators_[
            1].predict(grid_pts[['x', 'y']]), cmap='viridis')
ax4.set_title('Second Decision Tree')


# # Pipelines
# 
# We will use the idea of pipelines generically here to refer to the combination of steps that need to be performed to solve a problem. 

# In[26]:


get_ipython().run_cell_magic('file', 'pipe_utils.py', "from sklearn.preprocessing import FunctionTransformer\nimport numpy as np\nfrom skimage.filters import laplace, gaussian, median\nfrom skimage.util import montage as montage2d\nimport matplotlib.pyplot as plt\n\n\n\ndef display_data(in_ax, raw_data, show_hist):\n    if (raw_data.shape[0] == 1) and (len(raw_data.shape) == 4):\n        # reformat channels first\n        in_data = raw_data[0].swapaxes(0, 2).swapaxes(1, 2)\n    else:\n        in_data = np.squeeze(raw_data)\n    if len(in_data.shape) == 1:\n        if show_hist:\n            in_ax.hist(in_data)\n        else:\n            in_ax.plot(in_data, 'r.')\n    elif len(in_data.shape) == 2:\n        if show_hist:\n            for i in range(in_data.shape[1]):\n                in_ax.hist(in_data[:, i], label='Dim:{}'.format(i), alpha=0.5)\n            in_ax.legend()\n        else:\n            if in_data.shape[1] == 2:\n                in_ax.plot(in_data[:, 0], in_data[:, 1], 'r.')\n            else:\n                in_ax.plot(in_data, '.')\n    elif len(in_data.shape) == 3:\n        if show_hist:\n            in_ax.hist(in_data.ravel())\n        else:\n            n_stack = np.stack([(x-x.mean())/x.std() for x in in_data], 0)\n            in_ax.imshow(montage2d(n_stack))\n\n\ndef show_pipe(pipe, in_data, show_hist=False):\n    m_rows = np.ceil((len(pipe.steps)+1)/3).astype(int)\n    fig, t_axs = plt.subplots(m_rows, 3, figsize=(12, 5*m_rows))\n    m_axs = t_axs.flatten()\n    [c_ax.axis('off') for c_ax in m_axs]\n    last_data = in_data\n    for i, (c_ax, (step_name, step_op)) in enumerate(zip(m_axs, [('Input Data', None)]+pipe.steps), 1):\n        if step_op is not None:\n            try:\n                last_data = step_op.transform(last_data)\n            except AttributeError:\n                try:\n                    last_data = step_op.predict_proba(last_data)\n                except AttributeError:\n                    last_data = step_op.predict(last_data)\n\n        display_data(c_ax, last_data, show_hist)\n        c_ax.set_title('Step {} {}\\n{}'.format(i, last_data.shape, step_name))\n        c_ax.axis('on')\n\n\ndef flatten_func(x): return np.reshape(x, (np.shape(x)[0], -1))\n\n\nflatten_step = FunctionTransformer(flatten_func, validate=False)\n\n\ndef px_flatten_func(in_x):\n    if len(in_x.shape) == 2:\n        x = np.expand_dims(in_x, -1)\n    elif len(in_x.shape) == 3:\n        x = in_x\n    elif len(in_x.shape) == 4:\n        x = in_x\n    else:\n        raise ValueError(\n            'Cannot work with images with dimensions {}'.format(in_x.shape))\n    return np.reshape(x, (-1, np.shape(x)[-1]))\n\n\npx_flatten_step = FunctionTransformer(px_flatten_func, validate=False)\n\n\ndef add_filters(in_x, filt_func=[lambda x: gaussian(x, sigma=2),\n                                 lambda x: gaussian(\n                                     x, sigma=5)-gaussian(x, sigma=2),\n                                 lambda x: gaussian(x, sigma=8)-gaussian(x, sigma=5)]):\n    if len(in_x.shape) == 2:\n        x = np.expand_dims(np.expand_dims(in_x, 0), -1)\n    elif len(in_x.shape) == 3:\n        x = np.expand_dims(in_x, -1)\n    elif len(in_x.shape) == 4:\n        x = in_x\n    else:\n        raise ValueError(\n            'Cannot work with images with dimensions {}'.format(in_x.shape))\n    n_img, x_dim, y_dim, c_dim = x.shape\n    out_imgs = [x]\n    for c_filt in filt_func:\n        out_imgs += [np.stack([np.stack([c_filt(x[i, :, :, j])\n                                         for i in range(n_img)], 0)\n                               for j in range(c_dim)], -1)]\n\n    return np.concatenate(out_imgs, -1)\n\n\nfilter_step = FunctionTransformer(add_filters, validate=False)\n\n\ndef add_xy_coord(in_x, polar=False):\n    if len(in_x.shape) == 2:\n        x = np.expand_dims(np.expand_dims(in_x, 0), -1)\n    elif len(in_x.shape) == 3:\n        x = np.expand_dims(in_x, -1)\n    elif len(in_x.shape) == 4:\n        x = in_x\n    else:\n        raise ValueError(\n            'Cannot work with images with dimensions {}'.format(in_x.shape))\n    n_img, x_dim, y_dim, c_dim = x.shape\n\n    _, xx, yy, _ = np.meshgrid(np.arange(n_img),\n                               np.arange(x_dim),\n                               np.arange(y_dim),\n                               [1],\n                               indexing='ij')\n    if polar:\n        rr = np.sqrt(np.square(xx-xx.mean())+np.square(yy-yy.mean()))\n        th = np.arctan2(yy-yy.mean(), xx-xx.mean())\n        return np.concatenate([x, rr, th], -1)\n    else:\n        return np.concatenate([x, xx, yy], -1)\n\n\nxy_step = FunctionTransformer(add_xy_coord, validate=False)\npolar_step = FunctionTransformer(\n    lambda x: add_xy_coord(x, polar=True), validate=False)\n\n\ndef fit_img_pipe(in_pipe, in_x, in_y):\n    in_pipe.fit(in_x,\n                px_flatten_func(in_y)[:, 0])\n\n    def predict_func(new_x):\n        x_dim, y_dim = new_x.shape[0:2]\n        return in_pipe.predict(new_x).reshape((x_dim, y_dim, -1))\n    return predict_func")


# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')


blob_data, blob_labels = make_blobs(n_samples=100,
                                    random_state=2018)
test_pts = pd.DataFrame(blob_data, columns=['x', 'y'])
test_pts['group_id'] = blob_labels

plt.scatter(test_pts.x, test_pts.y, c=test_pts.group_id, cmap='viridis')
test_pts.sample(5)


# In[28]:


from pipe_utils import show_pipe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
simple_pipe = Pipeline([('Normalize', RobustScaler())])
simple_pipe.fit(test_pts)

show_pipe(simple_pipe, test_pts.values)
show_pipe(simple_pipe, test_pts.values, show_hist=True)


# In[29]:


from sklearn.preprocessing import QuantileTransformer
longer_pipe = Pipeline([('Quantile', QuantileTransformer(2)),
                        ('Normalize', RobustScaler())
                        ])
longer_pipe.fit(test_pts)

show_pipe(longer_pipe, test_pts.values)
show_pipe(longer_pipe, test_pts.values, show_hist=True)


# In[30]:


from sklearn.preprocessing import PolynomialFeatures
messy_pipe = Pipeline([
    ('Normalize', RobustScaler()),
    ('PolynomialFeatures', PolynomialFeatures(2)),
])
messy_pipe.fit(test_pts)

show_pipe(messy_pipe, test_pts.values)
show_pipe(messy_pipe, test_pts.values, show_hist=True)


# # Classification
# 
# A common problem of putting images into categories. The standard problem for this is classifying digits between 0 and 9. Fundamentally a classification problem is one where we are taking a large input (images, vectors, ...) and trying to put it into a category. 
# 
# 

# In[31]:


from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 
from pipe_utils import show_pipe
get_ipython().run_line_magic('matplotlib', 'inline')
digit_ds = load_digits(return_X_y=False)
img_data = digit_ds.images[:50]
digit_id = digit_ds.target[:50]
print('Image Data', img_data.shape)


# In[32]:


from pipe_utils import flatten_step
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
digit_pipe = Pipeline([('Flatten', flatten_step),
                       ('Normalize', RobustScaler())])
digit_pipe.fit(img_data)

show_pipe(digit_pipe, img_data)
show_pipe(digit_pipe, img_data, show_hist=True)


# In[33]:


from sklearn.neighbors import KNeighborsClassifier

digit_class_pipe = Pipeline([('Flatten', flatten_step),
                             ('Normalize', RobustScaler()),
                             ('NearestNeighbor', KNeighborsClassifier(1))])
digit_class_pipe.fit(img_data, digit_id)

show_pipe(digit_class_pipe, img_data)


# In[34]:


from sklearn.metrics import accuracy_score
pred_digit = digit_class_pipe.predict(img_data)
print('%2.2f%% accuracy' % (100*accuracy_score(digit_id, pred_digit)))


# In[35]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
fig, ax1 = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
sns.heatmap(
    confusion_matrix(digit_id, pred_digit),
    annot=True,
    fmt='d',
    ax=ax1)


# In[36]:


from sklearn.metrics import classification_report
print(classification_report(digit_id, pred_digit))


# # Wow! We've built an amazing algorithm!
# ## Let's patent it! Call Google!

# In[37]:


test_digit = np.array([[[0.,  0.,  6., 12., 13.,  6.,  0.,  0.],
                        [0.,  6., 16.,  9., 12., 16.,  2.,  0.],
                        [0.,  7., 16.,  9., 15., 13.,  0.,  0.],
                        [0.,  0., 11., 15., 16.,  4.,  0.,  0.],
                        [0.,  0.,  0., 12., 10.,  0.,  0.,  0.],
                        [0.,  0.,  3., 16.,  4.,  0.,  0.,  0.],
                        [0.,  0.,  1., 16.,  2.,  0.,  0.,  0.],
                        [0.,  0.,  6., 11.,  0.,  0.,  0.,  0.]]])
plt.matshow(test_digit[0], cmap='bone')
print('Prediction:', digit_class_pipe.predict(test_digit))
print('Real Value:', 9)


# # Training, Validation, and Testing
# 
# https://www.kdnuggets.com/2017/08/dataiku-predictive-model-holdout-cross-validation.html
# 
# ![image.png](attachment:image.png)
# 

# # Regression
# For regression, we can see it very similarly to a classification but instead of trying to output discrete classes we can output on a continuous scale. So we can take the exact same task (digits) but instead of predicting the category we can predict the actual decimal number

# In[38]:


from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pipe_utils import show_pipe, flatten_step
get_ipython().run_line_magic('matplotlib', 'inline')
digit_ds = load_digits(return_X_y=False)

img_data = digit_ds.images[:50]
digit_id = digit_ds.target[:50]

valid_data = digit_ds.images[50:500]
valid_id = digit_ds.target[50:500]


# In[39]:


from sklearn.neighbors import KNeighborsRegressor

digit_regress_pipe = Pipeline([('Flatten', flatten_step),
                               ('Normalize', RobustScaler()),
                               ('NearestNeighbor', KNeighborsRegressor(1))])
digit_regress_pipe.fit(img_data, digit_id)

show_pipe(digit_regress_pipe, img_data)


# # Assessment
# We can't use accuracy, ROC, precision, recall or any of these factors anymore since we don't have binary / true-or-false conditions we are trying to predict. We know have to go back to some of the initial metrics we covered in the first lectures.
# 
# $$ MSE = \frac{1}{N}\sum \left(y_{predicted} - y_{actual}\right)^2 $$
# $$ MAE = \frac{1}{N}\sum |y_{predicted} - y_{actual}| $$

# In[40]:


import seaborn as sns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
pred_train = digit_regress_pipe.predict(img_data)
jitter = lambda x: x+0.25*np.random.uniform(-1, 1, size=x.shape)
sns.swarmplot(digit_id, jitter(pred_train), ax=ax1)
ax1.set_title('Predictions (Training)\nMSE: %2.2f MAE: %2.2f' % (np.mean(np.square(pred_train-digit_id)),
                                                                 np.mean(np.abs(pred_train-digit_id))))

pred_valid = digit_regress_pipe.predict(valid_data)
sns.swarmplot(valid_id, jitter(pred_valid), ax=ax2)
ax2.set_title('Predictions (Validation)\nMSE: %2.2f MAE: %2.2f' % (np.mean(np.square(pred_valid-valid_id)),
                                                                   np.mean(np.abs(pred_valid-valid_id))))


# ## Increasing neighbor count

# In[41]:


digit_regress_pipe = Pipeline([('Flatten', flatten_step),
                               ('Normalize', RobustScaler()),
                               ('NearestNeighbor', KNeighborsRegressor(2))])
digit_regress_pipe.fit(img_data, digit_id)

show_pipe(digit_regress_pipe, img_data)


# In[42]:


import seaborn as sns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
pred_train = digit_regress_pipe.predict(img_data)

sns.swarmplot(digit_id, jitter(pred_train), ax=ax1)
ax1.set_title('Predictions (Training)\nMSE: %2.2f MAE: %2.2f' % (np.mean(np.square(pred_train-digit_id)),
                                                                 np.mean(np.abs(pred_train-digit_id))))

pred_valid = digit_regress_pipe.predict(valid_data)
sns.swarmplot(valid_id, jitter(pred_valid), ax=ax2)
ax2.set_title('Predictions (Validation)\nMSE: %2.2f MAE: %2.2f' % (np.mean(np.square(pred_valid-valid_id)),
                                                                   np.mean(np.abs(pred_valid-valid_id))))


# # Segmentation (Pixel Classification)
# 
# The first tasks we had were from one entire image to a single class (classification) or value (regression). Now we want to change problem, instead of a single class for each image, we want a class or value for each pixel. This requires that we restructure the problem.

# # Where segmentation fails: Mitochondria Segmentation in EM
# 
# ![Cortex Image](../common/data/em_image.png)
# 
# - The cortex is barely visible to the human eye
# - Tiny structures hint at where cortex is located
# 
# *** 
# 
# - A simple threshold is insufficient to finding the cortical structures
# - Other filtering techniques are unlikely to magicially fix this problem
# 

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
get_ipython().run_line_magic('matplotlib', 'inline')
cell_img = (imread("../common/data/em_image.png")[::2, ::2])/255.0
cell_seg = imread("../common/data/em_image_seg.png",
                  as_gray=True)[::2, ::2] > 0
print(cell_img.shape, cell_seg.shape)
np.random.seed(2018)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8), dpi=72)
ax1.imshow(cell_img, cmap='bone')
ax2.imshow(cell_seg, cmap='bone')


# In[44]:


train_img, valid_img = cell_img[:, :256], cell_img[:, 256:]
train_mask, valid_mask = cell_seg[:, :256], cell_seg[:, 256:]
print('Training', train_img.shape, train_mask.shape)
print('Validation Data', valid_img.shape, valid_mask.shape)


# In[45]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), dpi=72)
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')
ax2.imshow(train_mask, cmap='bone')
ax2.set_title('Train Mask')

ax3.imshow(valid_img, cmap='bone')
ax3.set_title('Validation Image')
ax4.imshow(valid_mask, cmap='bone')
ax4.set_title('Validation Mask')


# In[46]:


from pipe_utils import px_flatten_step, show_pipe, fit_img_pipe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

rf_seg_model = Pipeline([('Pixel Flatten', px_flatten_step),
                         ('Robust Scaling', RobustScaler()),
                         ('Decision Tree', DecisionTreeRegressor())
                         ])

pred_func = fit_img_pipe(rf_seg_model, train_img, train_mask)
show_pipe(rf_seg_model, train_img)
show_tree(rf_seg_model.steps[-1][1])


# In[47]:


fig, ((ax1, ax5, ax2), (ax3, ax6, ax4)) = plt.subplots(
    2, 3, figsize=(12, 8), dpi=72)
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')

ax5.imshow(train_mask, cmap='viridis')
ax5.set_title('Train Mask')

ax2.imshow(pred_func(train_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax2.set_title('Prediction Mask')

ax3.imshow(cell_img, cmap='bone')
ax3.set_title('Full Image')

ax6.imshow(cell_seg, cmap='viridis')
ax6.set_title('Full Mask')

ax4.imshow(pred_func(cell_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Prediction Mask')


# # Include Position Information

# In[48]:


from pipe_utils import xy_step

rf_xyseg_model = Pipeline([('Add XY', xy_step),
                           ('Pixel Flatten', px_flatten_step),
                           ('Normalize', RobustScaler()),
                           ('DecisionTree', DecisionTreeRegressor(
                               min_samples_split=1000))
                           ])

pred_func = fit_img_pipe(rf_xyseg_model, train_img, train_mask)
show_pipe(rf_xyseg_model, train_img)
show_tree(rf_xyseg_model.steps[-1][1])


# In[49]:


fig, ((ax1, ax5, ax2), (ax3, ax6, ax4)) = plt.subplots(
    2, 3, figsize=(12, 8), dpi=72)
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')

ax5.imshow(train_mask, cmap='viridis')
ax5.set_title('Train Mask')

ax2.imshow(pred_func(train_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax2.set_title('Prediction Mask')

ax3.imshow(cell_img, cmap='bone')
ax3.set_title('Full Image')

ax6.imshow(cell_seg, cmap='viridis')
ax6.set_title('Full Mask')

ax4.imshow(pred_func(cell_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Prediction Mask')


# In[50]:


from sklearn.cluster import KMeans
rf_xyseg_k_model = Pipeline([('Add XY', xy_step),
                             ('Pixel Flatten', px_flatten_step),
                             ('Normalize', RobustScaler()),
                             ('KMeans', KMeans(4)),
                             ('RandomForest', RandomForestRegressor(n_estimators=25))
                             ])

pred_func = fit_img_pipe(rf_xyseg_k_model, train_img, train_mask)
show_pipe(rf_xyseg_k_model, train_img)


# In[51]:


fig, ((ax1, ax5, ax2), (ax3, ax6, ax4)) = plt.subplots(
    2, 3, figsize=(12, 8), dpi=72)
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')

ax5.imshow(train_mask, cmap='viridis')
ax5.set_title('Train Mask')

ax2.imshow(pred_func(train_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax2.set_title('Prediction Mask')

ax3.imshow(cell_img, cmap='bone')
ax3.set_title('Full Image')

ax6.imshow(cell_seg, cmap='viridis')
ax6.set_title('Full Mask')

ax4.imshow(pred_func(cell_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Prediction Mask')


# In[52]:


from sklearn.preprocessing import PolynomialFeatures
rf_xyseg_py_model = Pipeline([('Add XY', xy_step),
                              ('Pixel Flatten', px_flatten_step),
                              ('Normalize', RobustScaler()),
                              ('Polynomial Features', PolynomialFeatures(2)),
                              ('RandomForest', RandomForestRegressor(n_estimators=25))
                              ])

pred_func = fit_img_pipe(rf_xyseg_py_model, train_img, train_mask)
show_pipe(rf_xyseg_py_model, train_img)


# In[53]:


fig, ((ax1, ax5, ax2), (ax3, ax6, ax4)) = plt.subplots(
    2, 3, figsize=(12, 8), dpi=72)
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')

ax5.imshow(train_mask, cmap='viridis')
ax5.set_title('Train Mask')

ax2.imshow(pred_func(train_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax2.set_title('Prediction Mask')

ax3.imshow(cell_img, cmap='bone')
ax3.set_title('Full Image')

ax6.imshow(cell_seg, cmap='viridis')
ax6.set_title('Full Mask')

ax4.imshow(pred_func(cell_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Prediction Mask')


# # Adding Smarter Features
# Here we add images with filters and gaussians

# In[54]:


from pipe_utils import filter_step
rf_filterseg_model = Pipeline([('Filters', filter_step),
                               ('Pixel Flatten', px_flatten_step),
                               ('Normalize', RobustScaler()),
                               ('RandomForest', RandomForestRegressor(n_estimators=25))
                               ])

pred_func = fit_img_pipe(rf_filterseg_model, train_img, train_mask)
show_pipe(rf_filterseg_model, train_img)


# In[55]:


fig, ((ax1, ax5, ax2), (ax3, ax6, ax4)) = plt.subplots(
    2, 3, figsize=(12, 8), dpi=72)
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')

ax5.imshow(train_mask, cmap='viridis')
ax5.set_title('Train Mask')

ax2.imshow(pred_func(train_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax2.set_title('Prediction Mask')

ax3.imshow(cell_img, cmap='bone')
ax3.set_title('Full Image')

ax6.imshow(cell_seg, cmap='viridis')
ax6.set_title('Full Mask')

ax4.imshow(pred_func(cell_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Prediction Mask')


# ## Using the Neighborhood
# We can also include the whole neighborhood by shifting the image in x and y. For the first example we will then use linear regression so we can see the exact coefficients that result.

# In[56]:


from sklearn.preprocessing import FunctionTransformer


def add_neighborhood(in_x, x_steps=3, y_steps=3):
    if len(in_x.shape) == 2:
        x = np.expand_dims(np.expand_dims(in_x, 0), -1)
    elif len(in_x.shape) == 3:
        x = np.expand_dims(in_x, -1)
    elif len(in_x.shape) == 4:
        x = in_x
    else:
        raise ValueError(
            'Cannot work with images with dimensions {}'.format(in_x.shape))
    n_img, x_dim, y_dim, c_dim = x.shape
    out_imgs = []
    for i in range(-x_steps, x_steps+1):
        for j in range(-y_steps, y_steps+1):
            out_imgs += [np.roll(np.roll(x,
                                         axis=1, shift=i),
                                 axis=2,
                                 shift=j)]
    return np.concatenate(out_imgs, -1)


def neighbor_step(x_steps=3, y_steps=3):
    return FunctionTransformer(
        lambda x: add_neighborhood(x, x_steps, y_steps),
        validate=False)


# In[57]:


linreg_neighborseg_model = Pipeline([('Neighbors', neighbor_step(1, 1)),
                               ('Pixel Flatten', px_flatten_step),
                               ('Linear Regression', LinearRegression())
                               ])

pred_func = fit_img_pipe(linreg_neighborseg_model, train_img, train_mask)
show_pipe(linreg_neighborseg_model, train_img)


# In[58]:


fig, ((ax1, ax5, ax2), (ax3, ax6, ax4)) = plt.subplots(
    2, 3, figsize=(12, 8), dpi=72)
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')

ax5.imshow(train_mask, cmap='viridis')
ax5.set_title('Train Mask')

ax2.imshow(pred_func(train_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax2.set_title('Prediction Mask')

ax3.imshow(cell_img, cmap='bone')
ax3.set_title('Full Image')

ax6.imshow(cell_seg, cmap='viridis')
ax6.set_title('Full Mask')

ax4.imshow(pred_func(cell_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Prediction Mask')


# ## Why Linear Regression?
# We choose linear regression so we could get easily understood coefficients. The model fits $\vec{m}$ and $b$ to the $\vec{x}_{i,j}$ points in the image $I(i,j)$ to match the $y_{i,j}$ output in the segmentation as closely as possible
# $$ y_{i,j} = \vec{m}\cdot\vec{x_{i,j}}+b $$
# For a 3x3 cases this looks like
# $$ \vec{x}_{i,j} = \left[I(i-1,j-1), I(i-1, j), I(i-1, j+1) \dots I(i+1,j-1), I(i+1, j), I(i+1, j+1)\right] $$

# In[59]:


m = linreg_neighborseg_model.steps[-1][1].coef_
b = linreg_neighborseg_model.steps[-1][1].intercept_
print('M:', m)
print('b:', b)


# ## Convolution
# The steps we have here make up a convolution and so what we have effectively done is use linear regression to learn which coefficients we should use in a convolutional kernel to get the best results

# In[60]:


from scipy.ndimage import convolve
m_mat = m.reshape((3, 3)).T
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
sns.heatmap(m_mat,
            annot=True,
            ax=ax1, fmt='2.2f',
            vmin=-m_mat.std(),
            vmax=m_mat.std())
ax1.set_title(r'Kernel $\vec{m}$')
ax2.imshow(cell_img)
ax2.set_title('Input Image')
ax2.axis('off')

ax3.imshow(convolve(cell_img, m_mat)+b,
           vmin=0,
           vmax=1,
           cmap='viridis')
ax3.set_title('Post Convolution Image')
ax3.axis('off')

ax4.imshow(pred_func(cell_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Predicted from Linear Model')
ax4.axis('off')


# ## Nearest Neighbor
# We can also use the neighborhood and nearest neighbor, this means for each pixel and its surrounds we find the pixel in the training set that looks most similar

# In[61]:


nn_neighborseg_model = Pipeline([('Neighbors', neighbor_step(1, 1)),
                               ('Pixel Flatten', px_flatten_step),
                               ('Normalize', RobustScaler()),
                               ('NearestNeighbor', KNeighborsRegressor(n_neighbors=1))
                               ])

pred_func = fit_img_pipe(nn_neighborseg_model, train_img, train_mask)
show_pipe(nn_neighborseg_model, train_img)


# In[62]:


fig, ((ax1, ax5, ax2), (ax3, ax6, ax4)) = plt.subplots(
    2, 3, figsize=(12, 8), dpi=72)
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')

ax5.imshow(train_mask, cmap='viridis')
ax5.set_title('Train Mask')

ax2.imshow(pred_func(train_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax2.set_title('Prediction Mask')

ax3.imshow(cell_img, cmap='bone')
ax3.set_title('Full Image')

ax6.imshow(cell_seg, cmap='viridis')
ax6.set_title('Full Mask')

ax4.imshow(pred_func(cell_img)[:, :, 0],
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Prediction Mask')


# # U-Net
# 
# The last approach we will briefly cover is the idea of [U-Net](https://arxiv.org/abs/1505.04597) a landmark paper from 2015 that dominates MICCAI submissions and contest winners today. A nice overview of the techniques is presented by [Vladimir Iglovikov](https://youtu.be/g6oIQ5MXBE4) a winner of a recent Kaggle competition on masking images of cars [slides](http://slides.com/vladimiriglovikov/kaggle-deep-learning-to-create-a-model-for-binary-segmentation-of-car-images)
# 
# ![U-Net Diagram](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

# In[63]:


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, concatenate
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
base_depth = 32
in_img = Input((None, None, 1), name='Image_Input')
lay_1 = Conv2D(base_depth, kernel_size=(3, 3), padding='same')(in_img)
lay_2 = Conv2D(base_depth, kernel_size=(3, 3), padding='same')(lay_1)
lay_3 = MaxPool2D((2, 2))(lay_2)
lay_4 = Conv2D(base_depth*2, kernel_size=(3, 3), padding='same')(lay_3)
lay_5 = Conv2D(base_depth*2, kernel_size=(3, 3), padding='same')(lay_4)
lay_6 = MaxPool2D((2, 2))(lay_5)
lay_7 = Conv2D(base_depth*4, kernel_size=(3, 3), padding='same')(lay_6)
lay_8 = Conv2D(base_depth*4, kernel_size=(3, 3), padding='same')(lay_7)
lay_9 = UpSampling2D((2, 2))(lay_8)
lay_10 = concatenate([lay_5, lay_9])
lay_11 = Conv2D(base_depth*2, kernel_size=(3, 3), padding='same')(lay_10)
lay_12 = Conv2D(base_depth*2, kernel_size=(3, 3), padding='same')(lay_11)
lay_13 = UpSampling2D((2, 2))(lay_12)
lay_14 = concatenate([lay_2, lay_13])
lay_15 = Conv2D(base_depth, kernel_size=(3, 3), padding='same')(lay_14)
lay_16 = Conv2D(base_depth, kernel_size=(3, 3), padding='same')(lay_15)
lay_17 = Conv2D(1, kernel_size=(1, 1), padding='same',
                activation='sigmoid')(lay_16)
t_unet = Model(inputs=[in_img], outputs=[lay_17], name='SmallUNET')
dot_mod = model_to_dot(t_unet, show_shapes=True, show_layer_names=False)
dot_mod.set_rankdir('UD')
SVG(dot_mod.create_svg())


# In[64]:


t_unet.summary()


# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
get_ipython().run_line_magic('matplotlib', 'inline')
cell_img = (imread("../common/data/em_image.png")[::2, ::2])/255.0
cell_seg = imread("../common/data/em_image_seg.png",
                  as_gray=True)[::2, ::2] > 0
train_img, valid_img = cell_img[:256, 50:250], cell_img[:, 256:]
train_mask, valid_mask = cell_seg[:256, 50:250], cell_seg[:, 256:]
# add channels and sample dimensions


def prep_img(x, n=1): return (
    prep_mask(x, n=n)-train_img.mean())/train_img.std()


def prep_mask(x, n=1): return np.stack([np.expand_dims(x, -1)]*n, 0)


print('Training', train_img.shape, train_mask.shape)
print('Validation Data', valid_img.shape, valid_mask.shape)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), dpi=72)
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')
ax2.imshow(train_mask, cmap='bone')
ax2.set_title('Train Mask')

ax3.imshow(valid_img, cmap='bone')
ax3.set_title('Validation Image')
ax4.imshow(valid_mask, cmap='bone')
ax4.set_title('Validation Mask')


# # Results from Untrained Model
# We can make predictions with an untrained model, but we clearly do not expect them to be very good

# In[66]:


fig, m_axs = plt.subplots(2, 3,
                          figsize=(18, 8), dpi=150)
for c_ax in m_axs.flatten():
    c_ax.axis('off')
((ax1, ax2, _), (ax3, ax4, ax5)) = m_axs
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')
ax2.imshow(train_mask, cmap='viridis')
ax2.set_title('Train Mask')

ax3.imshow(cell_seg, cmap='bone')
ax3.set_title('Full Image')

unet_pred = t_unet.predict(prep_img(cell_img))[0, :, :, 0]
ax4.imshow(unet_pred,
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Predicted Segmentation')

ax5.imshow(cell_seg,
           cmap='viridis')
ax5.set_title('Ground Truth')


# # Note
# This is a very bad way to train a model, the loss function is poorly chosen, the optimizer can be improved the learning rate can be changed, the training and validation data **should not** come from the same sample (and **definitely** not the same measurement). The goal is to be aware of these techniques and have a feeling for how they can work for complex problems 

# In[67]:


from keras.optimizers import SGD
t_unet.compile(
    # we use a simple loss metric of mean-squared error to optimize
    loss='mse',
    # we use stochastic gradient descent to optimize
    optimizer=SGD(lr=0.05),
    # we keep track of the number of pixels correctly classified and the mean absolute error as well
    metrics=['binary_accuracy', 'mae']
)

loss_history = t_unet.fit(prep_img(train_img, n=5),
                          prep_mask(train_mask, n=5),
                          validation_data=(prep_img(valid_img),
                                           prep_mask(valid_mask)),
                          epochs=20)


# In[68]:


fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(20, 7))
ax1.plot(loss_history.epoch,
         loss_history.history['mean_absolute_error'], 'r-', label='Training')
ax1.plot(loss_history.epoch,
         loss_history.history['val_mean_absolute_error'], 'b-', label='Validation')
ax1.set_title('Mean Absolute Error')
ax1.legend()

ax2.plot(loss_history.epoch,
         100*np.array(loss_history.history['binary_accuracy']), 'r-', label='Training')
ax2.plot(loss_history.epoch,
         100*np.array(loss_history.history['val_binary_accuracy']), 'b-', label='Validation')
ax2.set_title('Classification Accuracy (%)')
ax2.legend()


# In[69]:


fig, m_axs = plt.subplots(2, 3,
                          figsize=(18, 8), dpi=150)
for c_ax in m_axs.flatten():
    c_ax.axis('off')
((ax1, ax15, ax2), (ax3, ax4, ax5)) = m_axs
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')
ax15.imshow(t_unet.predict(prep_img(train_img))[0, :, :, 0],
            cmap='viridis', vmin=0, vmax=1)
ax15.set_title('Predicted Training')
ax2.imshow(train_mask, cmap='viridis')
ax2.set_title('Train Mask')

ax3.imshow(cell_img, cmap='bone')
ax3.set_title('Full Image')
unet_pred = t_unet.predict(prep_img(cell_img))[0, :, :, 0]
ax4.imshow(unet_pred,
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Predicted Segmentation')

ax5.imshow(cell_seg,
           cmap='viridis')
ax5.set_title('Ground Truth')


# ## Overfitting
# 
# Having a model with 470,000 free parameters means that it is quite easy to overfit the model by training for too long. Basically what this means is like before the model has gotten very good at the training data but hasn't generalized to other kinds of problems and thus starts to perform worse on regions that aren't exactly the same as the training

# In[75]:


t_unet.compile(
    # we use a simple loss metric of mean-squared error to optimize
    loss='mse',
    # we use stochastic gradient descent to optimize
    optimizer=SGD(lr=0.3),
    # we keep track of the number of pixels correctly classified and the mean absolute error as well
    metrics=['binary_accuracy', 'mae']
)

loss_history = t_unet.fit(prep_img(train_img),
                          prep_mask(train_mask),
                          validation_data=(prep_img(valid_img),
                                           prep_mask(valid_mask)),
                          epochs=5)


# In[76]:


fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(20, 7))
ax1.plot(loss_history.epoch,
         loss_history.history['mean_absolute_error'], 'r-', label='Training')
ax1.plot(loss_history.epoch,
         loss_history.history['val_mean_absolute_error'], 'b-', label='Validation')
ax1.set_title('Mean Absolute Error')
ax1.legend()

ax2.plot(loss_history.epoch,
         100*np.array(loss_history.history['binary_accuracy']), 'r-', label='Training')
ax2.plot(loss_history.epoch,
         100*np.array(loss_history.history['val_binary_accuracy']), 'b-', label='Validation')
ax2.set_title('Classification Accuracy (%)')
ax2.legend()


# In[79]:


fig, m_axs = plt.subplots(2, 3,
                          figsize=(18, 8), dpi=150)
for c_ax in m_axs.flatten():
    c_ax.axis('off')
((ax1, ax15, ax2), (ax3, ax4, ax5)) = m_axs
ax1.imshow(train_img, cmap='bone')
ax1.set_title('Train Image')
ax15.imshow(t_unet.predict(prep_img(train_img))[0, :, :, 0],
            cmap='viridis', vmin=0, vmax=1)
ax15.set_title('Predicted Training')
ax2.imshow(train_mask, cmap='viridis')
ax2.set_title('Train Mask')

ax3.imshow(cell_img, cmap='bone')
ax3.set_title('Full Image')
unet_pred = t_unet.predict(prep_img(cell_img))[0, :, :, 0]
ax4.imshow(unet_pred,
           cmap='viridis', vmin=0, vmax=1)
ax4.set_title('Predicted Segmentation')

ax5.imshow(cell_seg,
           cmap='viridis')
ax5.set_title('Ground Truth');


# In[ ]:




