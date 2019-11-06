#!/usr/bin/env python
# coding: utf-8

# # ETHZ: 227-0966-00L
# # Quantitative Big Imaging
# # March 28, 2019
#
# ## Shape Analysis

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.style.use("ggplot")
sns.set_style("whitegrid", {"axes.grid": False})


# # Literature / Useful References
#
# - Jean Claude, Morphometry with R
# - [Online](http://link.springer.com/book/10.1007%2F978-0-387-77789-4) through ETHZ
# - [Buy it](http://www.amazon.com/Morphometrics-R-Use-Julien-Claude/dp/038777789X)
# - John C. Russ, “The Image Processing Handbook”,(Boca Raton, CRC Press)
# - Available [online](http://dx.doi.org/10.1201/9780203881095) within domain ethz.ch (or proxy.ethz.ch / public VPN)
# - Principal Component Analysis
#  - Venables, W. N. and B. D. Ripley (2002). Modern Applied Statistics with S, Springer-Verlag
# - Shape Tensors
#  - http://www.cs.utah.edu/~gk/papers/vissym04/
#  - Doube, M.,et al. (2010). BoneJ: Free and extensible bone image analysis in ImageJ. Bone, 47, 1076–9. doi:10.1016/j.bone.2010.08.023
#  - Mader, K. , et al. (2013). A quantitative framework for the 3D characterization of the osteocyte lacunar system. Bone, 57(1), 142–154. doi:10.1016/j.bone.2013.06.026
#
#  - Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
#   Core Algorithms. Springer-Verlag, London, 2009.
#  -  B. Jähne. Digital Image Processing. Springer-Verlag,
#            Berlin-Heidelberg, 6. edition, 2005.
#  -  T. H. Reiss. Recognizing Planar Objects Using Invariant Image
#            Features, from Lecture notes in computer science, p. 676. Springer,
#            Berlin, 1993.
#  - http://en.wikipedia.org/wiki/Image_moment
#
#

# # Previously on QBI ...
#
# - Image Enhancment
#  - Highlighting the contrast of interest in images
#  - Minimizing Noise
# - Segmentation
#  - Understanding value histograms
#  - Dealing with multi-valued data
# - Automatic Methods
#  - Hysteresis Method, K-Means Analysis
# - Regions of Interest
#  - Contouring
# - Machine Learning

# # Learning Objectives
#
# ## Motivation (Why and How?)
# - How do we quantify where and how big our objects are?
# - How can we say something about the shape?
# - How can we compare objects of different sizes?
# - How can we compare two images on the basis of the shape as calculated from the images?
# - How can we put objects into an finite element simulation? or make pretty renderings?

# # Outline
#
# - Motivation (Why and How?)
# - Object Characterization
# - Volume
# - Center and Extents
# - Anisotropy
#
# ***
#
# - Shape Tensor
# - Principal Component Analysis
# - Ellipsoid Representation
# - Scale-free metrics
# - Anisotropy, Oblateness
# - Meshing
#  - Marching Cubes
#  - Isosurfaces
# - Surface Area

# # Motivation
#
#
# We have dramatically simplified our data, but there is still too much.
#
# - We perform an experiment bone to see how big the cells are inside the tissue
# $$\downarrow$$ ![Bone Measurement](ext-figures/tomoimage.png)
#
# ### 2560 x 2560 x 2160 x 32 bit
# _56GB / sample_
# - Filtering and Enhancement!
# $$\downarrow$$
# - 56GB of less noisy data
#
# ***
#
# - __Segmentation__
#
# $$\downarrow$$
#
# ### 2560 x 2560 x 2160 x 1 bit
# (1.75GB / sample)
#
# - Still an aweful lot of data

# # What did we want in the first place
#
# ### _Single number_:
# * volume fraction,
# * cell count,
# * average cell stretch,
# * cell volume variability

# # Component Labeling
#
# Once we have a clearly segmented image, it is often helpful to identify the sub-components of this image. The easist method for identifying these subcomponents is called component labeling which again uses the neighborhood $\mathcal{N}$ as a criterion for connectivity, resulting in pixels which are touching being part of the same object.
#
#
# In general, the approach works well since usually when different regions are touching, they are related. It runs into issues when you have multiple regions which agglomerate together, for example a continuous pore network (1 object) or a cluster of touching cells.
#
# Here we show some examples from Cityscape Data taken in Aachen (https://www.cityscapes-dataset.com/)

# In[2]:


from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")

car_img = imread("ext-figures/aachen_img.png")
seg_img = imread("ext-figures/aachen_label.png")[::4, ::4] == 26
print("image dimensions", car_img.shape, seg_img.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.imshow(car_img)
ax1.set_title("Input Image")

ax2.imshow(seg_img, cmap="bone")
ax2.set_title("Segmented Image")


# The more general formulation of the problem is for networks (roads, computers, social). Are the points start and finish connected?

# In[3]:


from skimage.morphology import label

help(label)


# In[4]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.imshow(seg_img, cmap="bone")
ax1.set_title("Segmented Image")
lab_img = label(seg_img)
ax2.imshow(lab_img, cmap=plt.cm.gist_earth)
ax2.set_title("Labeled Image")


# In[5]:


fig, (ax3) = plt.subplots(1, 1)
ax3.hist(lab_img.ravel())
ax3.set_title("Label Counts")
ax3.set_yscale("log")


# # Component Labeling: Algorithm
#
# We start off with all of the pixels in either foreground (1) or background (0)

# In[6]:


from skimage.morphology import label
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
seg_img = np.eye(9, dtype=int)
seg_img[4, 4] = 0
seg_img += seg_img[::-1]
sns.heatmap(seg_img, annot=True, fmt="d")


# Give each point in the image a unique label
# - For each point $(x,y)\in\text{Foreground}$
#  - Set value to $I_{x,y} = x+y*width+1$

# In[7]:


idx_img = np.zeros_like(seg_img)
for x in range(seg_img.shape[0]):
    for y in range(seg_img.shape[1]):
        if seg_img[x, y] > 0:
            idx_img[x, y] = x + y * seg_img.shape[0] + 1
sns.heatmap(idx_img, annot=True, fmt="d", cmap="nipy_spectral")


# In a [brushfire](http://www.sciencedirect.com/science/article/pii/S0921889007000966)-style algorithm
# - For each point $(x,y)\in\text{Foreground}$
#     - For each point $(x^{\prime},y^{\prime})\in\mathcal{N}(x,y)$
#     - if $(x^{\prime},y^{\prime})\in\text{Foreground}$
#         - Set the label to $\min(I_{x,y}, I_{x^{\prime},y^{\prime}})$
# - Repeat until no more labels have been changed

# In[8]:


fig, m_axs = plt.subplots(2, 2, figsize=(20, 20))
last_img = idx_img.copy()
img_list = [last_img]
for iteration, c_ax in enumerate(m_axs.flatten(), 1):
    cur_img = last_img.copy()

    for x in range(last_img.shape[0]):
        for y in range(last_img.shape[1]):
            if last_img[x, y] > 0:
                i_xy = last_img[x, y]
                for xp in [-1, 0, 1]:
                    if (x + xp < last_img.shape[0]) and (x + xp >= 0):
                        for yp in [-1, 0, 1]:
                            if (y + yp < last_img.shape[1]) and (y + yp >= 0):
                                i_xpyp = last_img[x + xp, y + yp]
                                if i_xpyp > 0:

                                    new_val = min(i_xy, i_xpyp, cur_img[x, y])
                                    if cur_img[x, y] != new_val:
                                        print(
                                            (x, y),
                                            i_xy,
                                            "vs",
                                            (x + xp, y + yp),
                                            i_xpyp,
                                            "->",
                                            new_val,
                                        )
                                        cur_img[x, y] = new_val

    img_list += [cur_img]
    sns.heatmap(cur_img, annot=True, fmt="d", cmap="nipy_spectral", ax=c_ax)
    c_ax.set_title("Iteration #{}".format(iteration))
    if (cur_img == last_img).all():
        print("Done")
        break
    else:
        print(
            "Iteration",
            iteration,
            "Groups",
            len(np.unique(cur_img[cur_img > 0].ravel())),
            "Changes",
            np.sum(cur_img != last_img),
        )
        last_img = cur_img


# The image very quickly converges and after 4 iterations the task is complete. For larger more complicated images with thousands of components this task can take longer, but there exist much more efficient [algorithms](https://www.cs.princeton.edu/~rs/AlgsDS07/01UnionFind.pdf) for labeling components which alleviate this issue.

# In[9]:


from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, c_ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)


def update_frame(i):
    plt.cla()
    sns.heatmap(
        img_list[i],
        annot=True,
        fmt="d",
        cmap="nipy_spectral",
        ax=c_ax,
        cbar=False,
        vmin=img_list[0].min(),
        vmax=img_list[0].max(),
    )
    c_ax.set_title(
        "Iteration #{}, Groups {}".format(
            i + 1, len(np.unique(img_list[i][img_list[i] > 0].ravel()))
        )
    )


# write animation frames
anim_code = FuncAnimation(
    fig, update_frame, frames=len(img_list) - 1, interval=1000, repeat_delay=2000
).to_html5_video()
plt.close("all")
HTML(anim_code)


# # Bigger Images
# How does the same algorithm apply to bigger images

# In[10]:


from skimage.io import imread
from skimage.morphology import label
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
seg_img = (imread("ext-figures/aachen_label.png")[::4, ::4] == 26)[110:130:2, 370:420:3]
seg_img[9, 1] = 1
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=150)
sns.heatmap(seg_img, annot=True, fmt="d", ax=ax1, cmap="nipy_spectral", cbar=False)
idx_img = seg_img * np.arange(len(seg_img.ravel())).reshape(seg_img.shape)
sns.heatmap(idx_img, annot=True, fmt="d", ax=ax2, cmap="nipy_spectral", cbar=False)


# In[11]:


last_img = idx_img.copy()
img_list = [last_img]
for iteration in range(99):
    cur_img = last_img.copy()
    for x in range(last_img.shape[0]):
        for y in range(last_img.shape[1]):
            if last_img[x, y] > 0:
                i_xy = last_img[x, y]
                for xp in [-1, 0, 1]:
                    if (x + xp < last_img.shape[0]) and (x + xp >= 0):
                        for yp in [-1, 0, 1]:
                            if (y + yp < last_img.shape[1]) and (y + yp >= 0):
                                i_xpyp = last_img[x + xp, y + yp]
                                if i_xpyp > 0:
                                    new_val = min(i_xy, i_xpyp, cur_img[x, y])
                                    if cur_img[x, y] != new_val:
                                        cur_img[x, y] = new_val

    img_list += [cur_img]
    if (cur_img == last_img).all():
        print("Done")
        break
    else:
        print(
            "Iteration",
            iteration,
            "Groups",
            len(np.unique(cur_img[cur_img > 0].ravel())),
            "Changes",
            np.sum(cur_img != last_img),
        )
        last_img = cur_img


# In[12]:


from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, c_ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)


def update_frame(i):
    plt.cla()
    sns.heatmap(
        img_list[i],
        annot=True,
        fmt="d",
        cmap="nipy_spectral",
        ax=c_ax,
        cbar=False,
        vmin=img_list[0].min(),
        vmax=img_list[0].max(),
    )
    c_ax.set_title(
        "Iteration #{}, Groups {}".format(
            i + 1, len(np.unique(img_list[i][img_list[i] > 0].ravel()))
        )
    )


# write animation frames
anim_code = FuncAnimation(
    fig, update_frame, frames=len(img_list) - 1, interval=500, repeat_delay=1000
).to_html5_video()
plt.close("all")
HTML(anim_code)


# # Different Neighborhoods
# We can expand beyond the 3x3 neighborhood to a 5x5 for example

# In[13]:


last_img = idx_img.copy()
img_list = [last_img]
for iteration in range(99):
    cur_img = last_img.copy()
    for x in range(last_img.shape[0]):
        for y in range(last_img.shape[1]):
            if last_img[x, y] > 0:
                i_xy = last_img[x, y]
                for xp in [-2, -1, 0, 1, 2]:
                    if (x + xp < last_img.shape[0]) and (x + xp >= 0):
                        for yp in [-2, -1, 0, 1, 2]:
                            if (y + yp < last_img.shape[1]) and (y + yp >= 0):
                                i_xpyp = last_img[x + xp, y + yp]
                                if i_xpyp > 0:
                                    new_val = min(i_xy, i_xpyp, cur_img[x, y])
                                    if cur_img[x, y] != new_val:
                                        cur_img[x, y] = new_val

    img_list += [cur_img]
    if (cur_img == last_img).all():
        print("Done")
        break
    else:
        print(
            "Iteration",
            iteration,
            "Groups",
            len(np.unique(cur_img[cur_img > 0].ravel())),
            "Changes",
            np.sum(cur_img != last_img),
        )
        last_img = cur_img

fig, c_ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)


def update_frame(i):
    plt.cla()
    sns.heatmap(
        img_list[i],
        annot=True,
        fmt="d",
        cmap="nipy_spectral",
        ax=c_ax,
        cbar=False,
        vmin=img_list[0].min(),
        vmax=img_list[0].max(),
    )
    c_ax.set_title(
        "Iteration #{}, Groups {}".format(
            i + 1, len(np.unique(img_list[i][img_list[i] > 0].ravel()))
        )
    )


# write animation frames
anim_code = FuncAnimation(
    fig, update_frame, frames=len(img_list) - 1, interval=500, repeat_delay=1000
).to_html5_video()
plt.close("all")
HTML(anim_code)


# # Or a smaller kernel
# By using a smaller kernel (in this case where $\sqrt{x^2+y^2}<=1$, we cause the number of iterations to fill to increase and prevent the last pixel from being grouped since it is only connected diagonally
#
# |   |   |   |
# |--:|--:|--:|
# |  0|  1|  0|
# |  1|  1|  1|
# |  0|  1|  0|
#

# In[14]:


last_img = idx_img.copy()
img_list = [last_img]
for iteration in range(99):
    cur_img = last_img.copy()
    for x in range(last_img.shape[0]):
        for y in range(last_img.shape[1]):
            if last_img[x, y] > 0:
                i_xy = last_img[x, y]
                for xp in [-1, 0, 1]:
                    if (x + xp < last_img.shape[0]) and (x + xp >= 0):
                        for yp in [-1, 0, 1]:
                            if np.abs(xp) + np.abs(yp) <= 1:
                                if (y + yp < last_img.shape[1]) and (y + yp >= 0):
                                    i_xpyp = last_img[x + xp, y + yp]
                                    if i_xpyp > 0:
                                        new_val = min(i_xy, i_xpyp, cur_img[x, y])
                                        if cur_img[x, y] != new_val:
                                            cur_img[x, y] = new_val

    img_list += [cur_img]
    if (cur_img == last_img).all():
        print("Done")
        break
    else:
        print(
            "Iteration",
            iteration,
            "Groups",
            len(np.unique(cur_img[cur_img > 0].ravel())),
            "Changes",
            np.sum(cur_img != last_img),
        )
        last_img = cur_img

fig, c_ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)


def update_frame(i):
    plt.cla()
    sns.heatmap(
        img_list[i],
        annot=True,
        fmt="d",
        cmap="nipy_spectral",
        ax=c_ax,
        cbar=False,
        vmin=img_list[0].min(),
        vmax=img_list[0].max(),
    )
    c_ax.set_title(
        "Iteration #{}, Groups {}".format(
            i + 1, len(np.unique(img_list[i][img_list[i] > 0].ravel()))
        )
    )


# write animation frames
anim_code = FuncAnimation(
    fig, update_frame, frames=len(img_list) - 1, interval=500, repeat_delay=1000
).to_html5_video()
plt.close("all")
HTML(anim_code)


# # Component Labeling: Beyond
#
#
# Now all the voxels which are connected have the same label. We can then perform simple metrics like
#
# - counting the number of voxels in each label to estimate volume.
# - looking at the change in volume during erosion or dilation to estimate surface area

# ### What we would like to to do
#
# - Count the cells
# - Say something about the cells
# - Compare the cells in this image to another image
# - But where do we start?
#
# # COV: With a single object
#
# $$ I_{id}(x,y) =
# \begin{cases}
# 1, & L(x,y) = id \\
# 0, & \text{otherwise}
# \end{cases}$$

# In[15]:


from IPython.display import Markdown
from skimage.io import imread
from skimage.morphology import label
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
seg_img = imread("ext-figures/aachen_label.png") == 26
seg_img = seg_img[::4, ::4]
seg_img = seg_img[110:130:2, 370:420:3]
seg_img[9, 1] = 1
lab_img = label(seg_img)
_, (ax1) = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
sns.heatmap(lab_img, annot=True, fmt="d", ax=ax1, cmap="nipy_spectral", cbar=False)


# ### Define a center
# $$ \bar{x} = \frac{1}{N} \sum_{\vec{v}\in I_{id}} \vec{v}\cdot\vec{i} $$
# $$ \bar{y} = \frac{1}{N} \sum_{\vec{v}\in I_{id}} \vec{v}\cdot\vec{j} $$
# $$ \bar{z} = \frac{1}{N} \sum_{\vec{v}\in I_{id}} \vec{v}\cdot\vec{k} $$
#

# In[16]:


x_coord, y_coord = [], []
for x in range(seg_img.shape[0]):
    for y in range(seg_img.shape[1]):
        if seg_img[x, y] == 1:
            x_coord += [x]
            y_coord += [y]
print("x,y coordinates", list(zip(x_coord, y_coord)))
Markdown("$\\bar{x} = %2.2f, \\bar{y} = %2.2f $" % (np.mean(x_coord), np.mean(y_coord)))


# # COM: With a single object
#
# If the gray values are kept (or other meaningful ones are used), this can be seen as a weighted center of volume or center of mass (using $I_{gy}$ to distinguish it from the labels)
#
# ### Define a center
# $$ \Sigma I_{gy} = \frac{1}{N} \sum_{\vec{v}\in I_{id}} I_{gy}(\vec{v}) $$
# $$ \bar{x} = \frac{1}{\Sigma I_{gy}} \sum_{\vec{v}\in I_{id}} (\vec{v}\cdot\vec{i}) I_{gy}(\vec{v}) $$
# $$ \bar{y} = \frac{1}{\Sigma I_{gy}} \sum_{\vec{v}\in I_{id}} (\vec{v}\cdot\vec{j}) I_{gy}(\vec{v}) $$
# $$ \bar{z} = \frac{1}{\Sigma I_{gy}} \sum_{\vec{v}\in I_{id}} (\vec{v}\cdot\vec{k}) I_{gy}(\vec{v}) $$
#

# In[17]:


from IPython.display import Markdown, display
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")


xx, yy = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
gray_img = 100 * (np.abs(xx * yy - 7) + np.square(yy - 4)) + 0.25
gray_img *= np.abs(xx - 5) < 3
gray_img *= np.abs(yy - 5) < 3
gray_img[gray_img > 0] += 5
seg_img = (gray_img > 0).astype(int)
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=150)

sns.heatmap(gray_img, ax=ax1, cmap="bone_r", cbar=True)
ax1.set_title("Intensity Image")

sns.heatmap(seg_img, ax=ax2, cmap="bone", cbar=False)
ax2.set_title("Segmented Image")


# In[18]:


x_coord, y_coord, i_val = [], [], []
for x in range(seg_img.shape[0]):
    for y in range(seg_img.shape[1]):
        if seg_img[x, y] == 1:
            x_coord += [x]
            y_coord += [y]
            i_val += [gray_img[x, y]]

x_coord = np.array(x_coord)
y_coord = np.array(y_coord)
i_val = np.array(i_val)
cov_x = np.mean(x_coord)
cov_y = np.mean(y_coord)

display(
    Markdown(
        """## Center of Volume: 
- $\\bar{x} = %2.2f$
- $\\bar{y} = %2.2f $"""
        % (cov_x, cov_y)
    )
)

com_x = np.sum(x_coord * i_val) / np.sum(i_val)
com_y = np.sum(y_coord * i_val) / np.sum(i_val)

display(
    Markdown(
        """## Center of Mass: 
- $\\bar{x}_m = %2.2f$
- $\\bar{y}_m = %2.2f $"""
        % (com_x, com_y)
    )
)

_, (ax1) = plt.subplots(1, 1, figsize=(7, 7), dpi=150)

ax1.matshow(gray_img, cmap="bone_r")
ax1.set_title("Intensity Image")
ax1.plot([cov_y], [cov_x], "ro", label="COV", markersize=20)
ax1.plot([com_y], [com_x], "bo", label="COM", markersize=20)
ax1.legend()


# In[19]:


from skimage.measure import regionprops

help(regionprops)


# In[20]:


from skimage.measure import regionprops

all_regs = regionprops(seg_img, intensity_image=gray_img)
for c_reg in all_regs:
    display(Markdown("# Region: {}".format(c_reg.label)))
    for k in dir(c_reg):
        if not k.startswith("_") and ("image" not in k):
            display(Markdown("- {} {}".format(k, getattr(c_reg, k))))


# # Extents: With a single object
#
# Exents or caliper lenghts are the size of the object in a given direction. Since the coordinates of our image our $x$ and $y$ the extents are calculated in these directions
#
# Define extents as the minimum and maximum values along the projection of the shape in each direction
# $$ \text{Ext}_x = \left\{ \forall \vec{v}\in I_{id}: max(\vec{v}\cdot\vec{i})-min(\vec{v}\cdot\vec{i})  \right\} $$
# $$ \text{Ext}_y = \left\{ \forall \vec{v}\in I_{id}: max(\vec{v}\cdot\vec{j})-min(\vec{v}\cdot\vec{j})  \right\} $$
# $$ \text{Ext}_z = \left\{ \forall \vec{}\in I_{id}: max(\vec{v}\cdot\vec{k})-min(\vec{v}\cdot\vec{k})  \right\} $$
#
# - Lots of information about each object now
# - But, I don't think a biologist has ever asked "How long is a cell in the $x$ direction? how about $y$?"

# In[21]:


from IPython.display import Markdown
from skimage.io import imread
from skimage.morphology import label
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
seg_img = imread("ext-figures/aachen_label.png") == 26
seg_img = seg_img[::4, ::4]
seg_img = seg_img[110:130:2, 378:420:3] > 0
seg_img = np.pad(seg_img, 3, mode="constant")
_, (ax1) = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
ax1.matshow(seg_img, cmap="bone_r")


# In[22]:


x_coord, y_coord = [], []
for x in range(seg_img.shape[0]):
    for y in range(seg_img.shape[1]):
        if seg_img[x, y] == 1:
            x_coord += [x]
            y_coord += [y]
xmin = np.min(x_coord)
xmax = np.max(x_coord)
ymin = np.min(y_coord)
ymax = np.max(y_coord)
print("X -> ", "Min:", xmin, "Max:", xmax)
print("Y -> ", "Min:", ymin, "Max:", ymax)


# In[23]:


from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

_, (ax1) = plt.subplots(1, 1, figsize=(7, 7), dpi=150)

ax1.matshow(seg_img, cmap="bone_r")

xw = xmax - xmin
yw = ymax - ymin

c_bbox = [Rectangle(xy=(ymin, xmin), width=yw, height=xw)]
c_bb_patch = PatchCollection(
    c_bbox, facecolor="none", edgecolor="red", linewidth=4, alpha=0.5
)
ax1.add_collection(c_bb_patch)


# # Concrete Example
# So how can we begin to apply the tools we have developed. We take the original car scene from before.

# In[24]:


from skimage.measure import regionprops, label
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")

car_img = np.clip(imread("ext-figures/aachen_img.png")[75:150] * 2.0, 0, 255).astype(
    np.uint8
)
lab_img = label(imread("ext-figures/aachen_label.png")[::4, ::4] == 26)[75:150]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
ax1.imshow(car_img)
ax1.set_title("Input Image")

plt.colorbar(ax2.imshow(lab_img, cmap="nipy_spectral"))
ax2.set_title("Labeled Image")


# # Shape Analysis
# We can perform shape analysis on the image and calculate basic shape parameters for each object

# In[25]:


from skimage.measure import regionprops
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# shape analysis
all_regions = regionprops(lab_img)

fig, ax1 = plt.subplots(1, 1, figsize=(12, 6), dpi=100)
ax1.imshow(car_img)
print("Found ", len(all_regions), "regions")
bbox_list = []
for c_reg in all_regions:
    ax1.plot(c_reg.centroid[1], c_reg.centroid[0], "o", markersize=5)
    bbox_list += [
        Rectangle(
            xy=(c_reg.bbox[1], c_reg.bbox[0]),
            width=c_reg.bbox[3] - c_reg.bbox[1],
            height=c_reg.bbox[2] - c_reg.bbox[0],
        )
    ]
c_bb_patch = PatchCollection(
    bbox_list, facecolor="none", edgecolor="red", linewidth=4, alpha=0.5
)
ax1.add_collection(c_bb_patch)


# # Statistics
# We can then generate a table full of these basic parameters for each object. In this case, we add color as an additional description

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
import webcolors
import pandas as pd
from skimage.morphology import erosion, disk


def ed_img(in_img):
    # shrink an image to a few pixels
    cur_img = in_img.copy()
    while cur_img.max() > 0:
        last_img = cur_img
        cur_img = erosion(cur_img, disk(1))
    return last_img


# guess color name based on rgb value
color_name_class = KNeighborsClassifier(1)
c_names = sorted(webcolors.css3_names_to_hex.keys())
color_name_class.fit([tuple(webcolors.name_to_rgb(k)) for k in c_names], c_names)


reg_df = pd.DataFrame(
    [
        dict(
            label=c_reg.label,
            bbox=c_reg.bbox,
            area=c_reg.area,
            centroid=c_reg.centroid,
            color=color_name_class.predict(
                np.mean(car_img[ed_img(lab_img == c_reg.label)], 0)[:3].reshape((1, -1))
            )[0],
        )
        for c_reg in all_regions
    ]
)
fig, m_axs = plt.subplots(len(all_regions), 1, figsize=(3, 14))
for c_ax, c_reg in zip(m_axs, all_regions):
    c_ax.imshow(car_img[c_reg.bbox[0] : c_reg.bbox[2], c_reg.bbox[1] : c_reg.bbox[3]])
    c_ax.axis("off")
    c_ax.set_title("Label {}".format(c_reg.label))
reg_df


# Anisotropy: What is it?
# ===
# By definition (New Oxford American): ```varying in magnitude according to the direction of measurement.```
#
# - It allows us to define metrics in respect to one another and thereby characterize shape.
# - Is it tall and skinny, short and fat, or perfectly round
#
# ***
#
# Due to its very vague definition, it can be mathematically characterized in many different very much unequal ways (in all cases 0 represents a sphere)
#
# $$ Aiso1 = \frac{\text{Longest Side}}{\text{Shortest Side}} - 1 $$
#
# $$ Aiso2 = \frac{\text{Longest Side}-\text{Shortest Side}}{\text{Longest Side}} $$
#
# $$ Aiso3 = \frac{\text{Longest Side}}{\text{Average Side Length}} - 1 $$
#
# $$ Aiso4 = \frac{\text{Longest Side}-\text{Shortest Side}}{\text{Average Side Length}} $$
#
# $$ \cdots \rightarrow \text{ ad nauseum} $$

# In[27]:


from collections import defaultdict
from skimage.measure import regionprops
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")

xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))


def side_len(c_reg):
    return sorted([c_reg.bbox[3] - c_reg.bbox[1], c_reg.bbox[2] - c_reg.bbox[0]])


aiso_funcs = [
    lambda x: side_len(x)[-1] / side_len(x)[0] - 1,
    lambda x: (side_len(x)[-1] - side_len(x)[0]) / side_len(x)[-1],
    lambda x: side_len(x)[-1] / np.mean(side_len(x)) - 1,
    lambda x: (side_len(x)[-1] - side_len(x)[0]) / np.mean(side_len(x)),
]


def ell_func(a, b):
    return np.sqrt(np.square(xx / a) + np.square(yy / b)) <= 1


# In[28]:


from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, m_axs = plt.subplots(2, 3, figsize=(12, 10), dpi=120)
ab_list = [
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (1.5, 5),
    (1, 5),
    (0.5, 5),
    (0.1, 5),
    (0.05, 5),
]
func_pts = defaultdict(list)


def update_frame(i):
    plt.cla()
    a, b = ab_list[i]
    c_img = ell_func(a, b)
    m_axs[0, 0].imshow(c_img, cmap="gist_earth")
    reg_info = regionprops(c_img.astype(int))[0]
    m_axs[0, 0].set_title("Shape #{}".format(i + 1))
    for j, (c_func, c_ax) in enumerate(zip(aiso_funcs, m_axs.flatten()[1:]), 1):
        func_pts[j] += [c_func(reg_info)]
        c_ax.plot(func_pts[j], "r-")
        c_ax.set_title("Anisotropy #{}".format(j))
        c_ax.set_ylim(-0.1, 3)
    m_axs.flatten()[-1].axis("off")


# write animation frames
anim_code = FuncAnimation(
    fig, update_frame, frames=len(ab_list) - 1, interval=500, repeat_delay=1000
).to_html5_video()
plt.close("all")
HTML(anim_code)


# # Useful Statistical Tools: Principal Component Analysis
#
# While many of the topics covered in Linear Algebra and Statistics courses might not seem very applicable to real problems at first glance, at least a few of them come in handy for dealing distributions of pixels _(they will only be briefly covered, for more detailed review look at some of the suggested material)_
#
# ### Principal Component Analysis
# Similar to K-Means insofar as we start with a series of points in a vector space and want to condense the information. With PCA instead of searching for distinct groups, we try to find a linear combination of components which best explain the variance in the system.
#
# ***
#
# As an example we will use a very simple example from spectroscopy

# In[29]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")

cm_dm = np.linspace(1000, 4000, 300)


def peak(cent, wid, h):
    return h / (wid * np.sqrt(2 * np.pi)) * np.exp(-np.square((cm_dm - cent) / wid))


def peaks(plist):
    return np.sum(
        np.stack([peak(cent, wid, h) for cent, wid, h in plist], 0), 0
    ) + np.random.uniform(0, 1, size=cm_dm.shape)


fat_curve = [(2900, 100, 500), (1680, 200, 400)]
protein_curve = [(2900, 50, 200), (3400, 100, 600), (1680, 200, 300)]
noise_curve = [(3000, 50, 1)]

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 6))

ax1.plot(cm_dm, peaks(fat_curve))
ax1.set_title("Fat IR Spectra")

ax2.plot(cm_dm, peaks(protein_curve))
ax2.set_title("Protein IR Spectra")

ax0.plot(cm_dm, peaks(noise_curve))
ax0.set_title("Noise IR Spectra")

ax0.set_ylim(ax2.get_ylim())
ax2.set_ylim(ax2.get_ylim())

pd.DataFrame({"cm^(-1)": cm_dm, "intensity": peaks(protein_curve)}).head(10)


# # Test Dataset of a number of curves
# We want to sort cells or samples into groups of being more fat like or more protein like.
#
# ## How can we analyze this data without specifically looking for peaks or building models?

# In[30]:


test_data = np.stack(
    [
        peaks(c_curve)
        for _ in range(20)
        for c_curve in [protein_curve, fat_curve, noise_curve]
    ],
    0,
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(test_data[:4].T, ".-")
ax1.legend(["Curve 1", "Curve 2", "Curve 3", "Curve 4"])
ax2.scatter(
    test_data[:, 0],
    test_data[:, 1],
    c=range(test_data.shape[0]),
    s=20,
    cmap="nipy_spectral",
)


# In[31]:


from sklearn.decomposition import PCA

pca_tool = PCA(5)
pca_tool.fit(test_data)


# # Useful Statistical Tools: Principal Component Analysis
#
# The first principal component provides
#
# The second principal component is then related to the unique information seperating chicken from corn prices but neither indices directly themselves (maybe the cost of antibiotics)

# In[32]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
score_matrix = pca_tool.transform(test_data)
ax1.plot(cm_dm, pca_tool.components_[0, :], label="Component #1")
ax1.plot(
    cm_dm,
    pca_tool.components_[1, :],
    label="Component #2",
    alpha=pca_tool.explained_variance_ratio_[0],
)
ax1.plot(
    cm_dm,
    pca_tool.components_[2, :],
    label="Component #3",
    alpha=pca_tool.explained_variance_ratio_[1],
)
ax1.legend()
ax2.scatter(score_matrix[:, 0], score_matrix[:, 1])
ax2.set_xlabel("Component 1")
ax2.set_ylabel("Component 2")


# In[33]:


fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), dpi=120)
ax1.bar(
    x=range(pca_tool.explained_variance_ratio_.shape[0]),
    height=100 * pca_tool.explained_variance_ratio_,
)
ax1.set_xlabel("Components")
ax1.set_ylabel("Explained Variance (%)")


# # Principal Component Analysis
# ## scikit-learn [Face Analyis](http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html)
#
# Here we show a more imaging related example from the scikit-learn documentation where we do basic face analysis with scikit-learn.
#

# In[34]:


from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition

# Load faces data
try:
    dataset = fetch_olivetti_faces(shuffle=True, random_state=2018, data_home=".")
    faces = dataset.data
except Exception as e:
    print("Face data not available", e)
    faces = np.random.uniform(0, 1, (400, 4096))

n_samples, n_features = faces.shape
n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


# In[35]:


def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2.0 * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(
            comp.reshape(image_shape),
            cmap=plt.cm.gray,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.0)


# #############################################################################
# List of the different estimators, whether to center and transpose the
# problem, and whether the transformer uses the clustering API.
estimators = [
    (
        "Eigenfaces - PCA using randomized SVD",
        decomposition.PCA(
            n_components=n_components, svd_solver="randomized", whiten=True
        ),
        True,
    )
]
# #############################################################################
# Plot a sample of the input data

plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

# #############################################################################
# Do the estimation and plot it

for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    data = faces
    if center:
        data = faces_centered
    estimator.fit(data)

    if hasattr(estimator, "cluster_centers_"):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_
    plot_gallery(name, components_[:n_components])

plt.show()


# # Applied PCA: Shape Tensor
#
# ## How do these statistical analyses help us?
# Going back to a single cell, we have the a distribution of $x$ and $y$ values.
# - are not however completely independent
# - greatest variance does not normally lie in either x nor y alone.
#
# A principal component analysis of the voxel positions, will calculate two new principal components (the components themselves are the relationships between the input variables and the scores are the final values.)
# - An optimal rotation of the coordinate system

# We start off by calculating the covariance matrix from the list of $x$, $y$, and $z$ points that make up our object of interest.
#
# $$ COV(I_{id}) = \frac{1}{N} \sum_{\forall\vec{v}\in I_{id}} \begin{bmatrix}
# \vec{v}_x\vec{v}_x & \vec{v}_x\vec{v}_y & \vec{v}_x\vec{v}_z\\
# \vec{v}_y\vec{v}_x & \vec{v}_y\vec{v}_y & \vec{v}_y\vec{v}_z\\
# \vec{v}_z\vec{v}_x & \vec{v}_z\vec{v}_y & \vec{v}_z\vec{v}_z
# \end{bmatrix} $$
#
# We then take the eigentransform of this array to obtain the eigenvectors (principal components, $\vec{\Lambda}_{1\cdots 3}$) and eigenvalues (scores, $\lambda_{1\cdots 3}$)
#
# $$ COV(I_{id}) \longrightarrow \underbrace{\begin{bmatrix}
# \vec{\Lambda}_{1x} & \vec{\Lambda}_{1y} & \vec{\Lambda}_{1z} \\
# \vec{\Lambda}_{2x} & \vec{\Lambda}_{2y} & \vec{\Lambda}_{2z} \\
# \vec{\Lambda}_{3x} & \vec{\Lambda}_{3y} & \vec{\Lambda}_{3z}
# \end{bmatrix}}_{\textrm{Eigenvectors}} * \underbrace{\begin{bmatrix}
# \lambda_1 & 0 & 0 \\
# 0 & \lambda_2 & 0 \\
# 0 & 0 & \lambda_3
# \end{bmatrix}}_{\textrm{Eigenvalues}} * \underbrace{\begin{bmatrix}
# \vec{\Lambda}_{1x} & \vec{\Lambda}_{1y} & \vec{\Lambda}_{1z} \\
# \vec{\Lambda}_{2x} & \vec{\Lambda}_{2y} & \vec{\Lambda}_{2z} \\
# \vec{\Lambda}_{3x} & \vec{\Lambda}_{3y} & \vec{\Lambda}_{3z}
# \end{bmatrix}^{T}}_{\textrm{Eigenvectors}} $$
# The principal components tell us about the orientation of the object and the scores tell us about the corresponding magnitude (or length) in that direction.

# In[36]:


from IPython.display import Markdown
from skimage.io import imread
from skimage.morphology import label
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
seg_img = imread("ext-figures/aachen_label.png") == 26
seg_img = seg_img[::4, ::4]
seg_img = seg_img[110:130:2, 378:420:3] > 0
seg_img = np.pad(seg_img, 3, mode="constant")
seg_img[0, 0] = 0
_, (ax1) = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
ax1.matshow(seg_img, cmap="bone_r")


# In[37]:


from sklearn.decomposition import PCA

x_coord, y_coord = np.where(seg_img > 0)
xy_pts = np.stack([x_coord, y_coord], 1)
shape_pca = PCA()
shape_pca.fit(xy_pts)
pca_xy_vals = shape_pca.transform(xy_pts)
_, (ax1) = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
ax1.plot(pca_xy_vals[:, 0], pca_xy_vals[:, 1], "rs", markersize=10)


# In[38]:


_, (ax1) = plt.subplots(1, 1, figsize=(7, 7), dpi=150)


ax1.plot(
    xy_pts[:, 0] - np.mean(xy_pts[:, 0]),
    xy_pts[:, 1] - np.mean(xy_pts[:, 1]),
    "rs",
    label="Points",
)
ax1.plot(
    [0, shape_pca.explained_variance_[0] / 2 * shape_pca.components_[0, 0]],
    [0, shape_pca.explained_variance_[0] / 2 * shape_pca.components_[0, 1]],
    "b-",
    label="PCA1",
)
ax1.plot(
    [0, shape_pca.explained_variance_[1] / 2 * shape_pca.components_[1, 0]],
    [0, shape_pca.explained_variance_[1] / 2 * shape_pca.components_[1, 1]],
    "g-",
    label="PCA2",
)
ax1.legend()


# # Principal Component Analysis: Take home message
#
# - We calculate the statistical distribution individually for $x$, $y$, and $z$ and the 'correlations' between them.
# - From these values we can estimate the orientation in the direction of largest variance
# - We can also estimate magnitude
# - These functions are implemented as ```princomp``` or ```pca``` in various languages and scale well to very large datasets.

# # Principal Component Analysis: Elliptical Model
#
#
# While the eigenvalues and eigenvectors are in their own right useful
# - Not obvious how to visually represent these tensor objects
# - Ellipsoidal (Ellipse in 2D) representation alleviates this issue
#
# ### Ellipsoidal Representation
# 1. Center of Volume is calculated normally
# 1. Eigenvectors represent the unit vectors for the semiaxes of the ellipsoid
# 1. $\sqrt{\text{Eigenvalues}}$ is proportional to the length of the semiaxis ($\mathcal{l}=\sqrt{5\lambda_i}$), derivation similar to moment of inertia tensor for ellipsoids.
#
# ***

# # Meshing
#
#
# Constructing a mesh for an image provides very different information than the image data itself. Most crucially this comes when looking at physical processes like deformation.
#
# While the images are helpful for visualizing we rarely have models for quantifying how difficult it is to turn a pixel __off__
#
# If the image is turned into a mesh we now have a list of vertices and edges. For these vertices and edges we can define forces. For example when looking at stress-strain relationships in mechanics using Hooke's Model
# $$ \vec{F}=k (\vec{x}_0-\vec{x}) $$
# the force needed to stretch one of these edges is proportional to how far it is stretched.

# # Meshing
#
#
# Since we uses voxels to image and identify the volume we can use the voxels themselves as an approimation for the surface of the structure.
# - Each 'exposed' face of a voxel belongs to the surface
#
# From this we can create a mesh by
#
# - adding each exposed voxel face to a list of surface squares.
# - adding connectivity information for the different squares (shared edges and vertices)
#
# A wide variety of methods of which we will only graze the surface (http://en.wikipedia.org/wiki/Image-based_meshing)

# # Marching Cubes
#
# ### Why
# Voxels are very poor approximations for the surface and are very rough (they are either normal to the x, y, or z axis and nothing between). Because of their inherently orthogonal surface normals, any analysis which utilizes the surface normal to calculate another value (growth, curvature, etc) is going to be very inaccurate at best and very wrong at worst.
#
# ### [How](https://en.wikipedia.org/wiki/Marching_cubes)
# The image is processed one voxel at a time and the neighborhood (not quite the same is the morphological definition) is checked at every voxel. From this configuration of values, faces are added to the mesh to incorporate the most simple surface which would explain the values.
#
# [Marching tetrahedra](http://en.wikipedia.org/wiki/Marching_tetrahedra) is for some applications a better suited approach

# # Next Time on QBI
#
#
# So while bounding box and ellipse-based models are useful for many object and cells, they do a very poor job with other samples
#
#
# ***
#
# ### Why
# - We assume an entity consists of connected pixels (wrong)
# - We assume the objects are well modeled by an ellipse (also wrong)
#
# ### What to do?
#
# - Is it 3 connected objects which should all be analzed seperately?
# - If we could __divide it__, we could then analyze each spart as an ellipse
# - Is it one network of objects and we want to know about the constrictions?
# - Is it a cell or organelle with docking sites for cell?
# - Neither extents nor anisotropy are very meaningful, we need a __more specific metric__ which can characterize

# In[ ]:
