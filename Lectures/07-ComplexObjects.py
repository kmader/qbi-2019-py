#!/usr/bin/env python
# coding: utf-8

# # ETHZ: 227-0966-00L
# # Quantitative Big Imaging
# # April 4, 2019
# 
# ## Complex Objects and Distributions

# # Literature / Useful References
# 
# ### Books
# - Jean Claude, Morphometry with R
# - [Online](http://link.springer.com/book/10.1007%2F978-0-387-77789-4) through ETHZ
# - [Buy it](http://www.amazon.com/Morphometrics-R-Use-Julien-Claude/dp/038777789X)
# - John C. Russ, “The Image Processing Handbook”,(Boca Raton, CRC Press)
# - Available [online](http://dx.doi.org/10.1201/9780203881095) within domain ethz.ch (or proxy.ethz.ch / public VPN) 
# - J. Weickert, Visualization and Processing of Tensor Fields
#  - [Online](http://books.google.ch/books?id=ScLxPORMob4C&lpg=PA220&ots=mYIeQbaVXP&dq=&pg=PA220#v=onepage&q&f=false)

# ### Papers / Sites
# - Voronoi Tesselations
#  - Ghosh, S. (1997). Tessellation-based computational methods for the characterization and analysis of heterogeneous microstructures. Composites Science and Technology, 57(9-10), 1187–1210
#  - [Wolfram Explanation](http://mathworld.wolfram.com/VoronoiDiagram.html)
# 
# - Self-Avoiding / Nearest Neighbor
#  - Schwarz, H., & Exner, H. E. (1983). The characterization of the arrangement of feature centroids in planes and volumes. Journal of Microscopy, 129(2), 155–169.
#  - Kubitscheck, U. et al. (1996). Single nuclear pores visualized by confocal microscopy and image processing. Biophysical Journal, 70(5), 2067–77.
# 
# - Alignment / Distribution Tensor
#  - Mader, K. et al (2013). A quantitative framework for the 3D characterization of the osteocyte lacunar system. Bone, 57(1), 142–154
#  - Aubouy, M., et al. (2003). A texture tensor to quantify deformations. Granular Matter, 5, 67–70. Retrieved from http://arxiv.org/abs/cond-mat/0301018
# - Two point correlation
#  - Dinis, L., et. al. (2007). Analysis of 3D solids using the natural neighbour radial point interpolation method. Computer Methods in Applied Mechanics and Engineering, 196(13-16)
#  

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


# # Previously on QBI ...
# 
# 
# - Image Enhancment 
#  - Highlighting the contrast of interest in images
#  - Minimizing Noise
# - Understanding image histograms
# - Automatic Methods
# - Component Labeling
# - Single Shape Analysis
# - Complicated Shapes (Thickness Maps)

# # Outline
# 
# - Motivation (Why and How?)
# - Skeletons
#  - Tortuosity
# - Watershed Segmentation
#  - Connected Objects

# ### Local Environment
#  - Neighbors
#  - Voronoi Tesselation
#  - Distribution Tensor
# 
# 
# ### Global Enviroment
#  - Alignment
#  - Self-Avoidance
#  - Two Point Correlation Function

# # Metrics
# 
# We examine a number of different metrics in this lecture and additionally to classifying them as Local and Global we can define them as point and voxel-based operations. 
# 
# ### Point Operations
# - Nearest Neighbor
# - Delaunay Triangulation
#   - Distribution Tensor
# - Point (Center of Volume)-based Voronoi Tesselation
# - Alignment
# 
# ### Voxel Operation
# - Voronoi Tesselation
# - Neighbor Counting
# - 2-point (N-point) correlation function

# # Learning Objectives
# 
# ### Motivation (Why and How?)
# 
# - How can we extract topology of a structure?
# - How do we identify seperate objects when they are connected?
# - How can we compare shape of complex objects when they grow?

# In[2]:


from skimage.morphology import binary_opening, binary_closing, disk
from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
get_ipython().run_line_magic('matplotlib', 'inline')
bw_img = imread("ext-figures/bonegfiltslice.png")[::2, ::2]
thresh_img = binary_closing(binary_opening(bw_img < 90, disk(1)), disk(2))
fg_dmap = distance_transform_edt(thresh_img)
bg_dmap = distance_transform_edt(1-thresh_img)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6), dpi=100)
ax1.imshow(bw_img, cmap='bone')
ax2.imshow(thresh_img, cmap='bone')
ax3.set_title('Segmentation')
ax3.imshow(fg_dmap, cmap='nipy_spectral')
ax3.set_title('Distance Map\nForeground')
ax4.imshow(bg_dmap, cmap='nipy_spectral')
ax4.set_title('Distance Map\nBackground')


# ### Distribution Objectives
# 1. We want to know how many cells are alive
#  - Maybe small cells are dead and larger cells are alive $\rightarrow$ examine the volume distribution
#  - Maybe living cells are round and dead cells are really spiky and pointy $\rightarrow$ examine anisotropy
# 
# 1. We want to know where the cells are alive or most densely packed
#  - We can visually inspect the sample (maybe even color by volume)
#  - We can examine the raw positions (x,y,z) but what does that really tell us?
#  - We can make boxes and count the cells inside each one
#  - How do we compare two regions in the same sample or even two samples?

# # So what do we still need
# 
# 
# 1. A way for counting cells in a region and estimating density without creating arbitrary boxes
# 1. A way for finding out how many cells are _near_ a given cell, it's nearest neighbors
# 1. A way for quantifying how far apart cells are and then comparing different regions within a sample
# 1. A way for quantifying and comparing orientations
# 
# 
# 
# ### What would be really great? 
# 
# A tool which could be adapted to answering a large variety of problems
# - multiple types of structures
# - multiple phases

# Destructive Measurements
# ===
# With most imaging techniques and sample types, the task of measurement itself impacts the sample.
# - Even techniques like X-ray tomography which _claim_ to be non-destructive still impart significant to lethal doses of X-ray radition for high resolution imaging
# - Electron microscopy, auto-tome-based methods, histology are all markedly more destructive and make longitudinal studies impossible
# - Even when such measurements are possible
#  - Registration can be a difficult task and introduce artifacts
# 
# 
# 
# ### Why is this important?
# 
# - techniques which allow us to compare different samples of the same type.
# - are sensitive to common transformations
#  - Sample B after the treatment looks like Sample A stretched to be 2x larger
#  - The volume fraction at the center is higher than the edges but organization remains the same

# # Skeletonization / Networks
# 
# 
# For some structures like cellular materials and trabecular bone, we want a more detailed analysis than just thickness. We want to know
# 
# - which structures are connected
# - how they are connected
# - express the network in a simple manner
#  - quantify tortuosity
#  - branching
#  
# We start with a simpler example from the EPFL Dataset: EPFL CVLab's Library of Tree-Reconstruction Examples (http://cvlab.epfl.ch/data/delin)

# In[3]:


import matplotlib.pyplot as plt  # for showing plots
from skimage.io import imread  # for reading images
import numpy as np  # for matrix operations and array support
import pandas as pd  # for reading the swc files (tables of somesort)


def read_swc(in_path):
    swc_df = pd.read_csv(in_path, sep=' ', comment='#',
                         header=None)
    # a pure guess here
    swc_df.columns = ['id', 'junk1', 'x', 'y', 'junk2', 'width', 'next_idx']
    return swc_df[['x', 'y', 'width']]


im_data = imread('ext-figures/ny_7.tif')
mk_data = read_swc('ext-figures/ny_7.swc')

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6))
ax1.imshow(im_data)
ax1.set_title('Aerial Image')
ax3.imshow(im_data, cmap='bone')
ax3.scatter(mk_data['x'], mk_data['y'], s=mk_data['width'], alpha=0.5)
ax3.set_title('Roads')


# In[4]:


im_crop = im_data[250:420:1, 170:280:1]
mk_crop = mk_data.query('y>250').query(
    'y<420').query('x>170').query('x<280').copy()
mk_crop.x = (mk_crop.x-170)/1
mk_crop.y = (mk_crop.y-250)/1
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6))
ax1.imshow(im_crop)
ax1.set_title('Aerial Image')
ax3.imshow(im_crop, cmap='bone')
ax3.scatter(mk_crop['x'], mk_crop['y'], s=mk_crop['width'], alpha=0.25)
ax3.set_title('Roads')


# In[5]:


import seaborn as sns
from skimage.morphology import opening, closing, disk  # for removing small objects
from skimage.color import rgb2hsv
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))


def thresh_image(in_img):
    v_img = rgb2hsv(in_img)[:, :, 2]
    th_img = v_img > 0.4
    op_img = opening(th_img, disk(1))
    return op_img


ax1.imshow(im_crop)
ax1.set_title('Aerial Image')
ax2.imshow(rgb2hsv(im_crop)[:, :, 2],
           cmap='bone')
ax2.set_title('HSV Representation')
seg_img = thresh_image(im_crop)
ax3.imshow(seg_img, cmap='bone')
ax3.set_title('Segmentation')


# In[6]:


import seaborn as sns
from skimage.morphology import label
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

ax1.imshow(im_crop)
ax1.set_title('Aerial Image')
lab_img = label(seg_img)
ax2.imshow(lab_img, cmap='gist_earth')
ax2.set_title('Labeling')

sns.heatmap(lab_img[::4, ::4],
            annot=True,
            fmt="d",
            cmap='gist_earth',
            ax=ax3,
            cbar=False,
            vmin=0,
            vmax=lab_img.max(),
            annot_kws={"size": 10})
ax3.set_title('Labels')


# # Skeletonization
# 
# The first step is to take the distance transform the structure 
# $$I_d(x,y) = \textrm{dist}(I(x,y))$$
# We can see in this image there are already local maxima that form a sort of backbone which closely maps to what we are interested in.
# 

# In[7]:


from scipy import ndimage
keep_lab_img = lab_img == 1
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
ax1.imshow(keep_lab_img)
ax1.set_title('Road Component\n(Largest)')
dist_map = ndimage.distance_transform_edt(keep_lab_img)
ax2.imshow(dist_map, cmap='nipy_spectral')
ax2.set_title('Distance Map')

sns.heatmap(dist_map[::4, ::4],
            annot=True,
            fmt="1.0f",
            cmap='nipy_spectral',
            ax=ax3,
            cbar=False,
            vmin=0,
            vmax=dist_map.max(),
            annot_kws={"size": 10})
ax3.set_title('Distance Map')


# # Skeletonization: Ridges
# 
# By using the laplacian filter as an approximate for the derivative operator which finds the values which high local gradients.
# 
# $$ \nabla I_{d}(x,y) = (\frac{\delta^2}{\delta x^2}+\frac{\delta^2}{\delta y^2})I_d \approx \underbrace{\begin{bmatrix}
# -1 & -1 & -1 \\
# -1 & 8 & -1 \\
# -1 & -1 & -1
# \end{bmatrix}}_{\textrm{Laplacian Kernel}} \otimes I_d(x,y) $$
# 
# ## Creating the skeleton
# 
# 
# We can locate the local maxima of the structure by setting a minimum surface distance
# $$I_d(x,y)>MIN_{DIST}$$
# and combining it with a minimum slope value 
# $$\nabla I_d(x,y) > MIN_{SLOPE}$$
# 
# ***
# 
# ### Thresholds
# Harking back to our earlier lectures, this can be seen as a threshold on a feature vector representation of the entire dataset.
# - We first make the dataset into a tuple
# 
# $$ \textrm{cImg}(x,y) = \langle \underbrace{I_d(x,y)}_1, \underbrace{\nabla I_d(x,y)}_2 \rangle $$
# 
# $$ \textrm{skelImage}(x,y) = $$
# $$ \begin{cases}
# 1, & \textrm{cImg}_1(x,y)\geq MIN-DIST \\ 
#  &  \& \textrm{ cImg}_2(x,y)\geq MIN-SLOPE \\
# 0, & \textrm{otherwise}
# \end{cases}$$ 
# 

# In[8]:


# for finding the medial axis and making skeletons
from skimage.morphology import medial_axis
# for just the skeleton code
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.filters import laplace

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

ax1.imshow(dist_map, cmap='nipy_spectral')
ax1.set_title('Distance Map')

ax2.imshow(laplace(dist_map), cmap='RdBu')
ax2.set_title('Laplacian of Distance')

# we use medial axis since it is cleaner
skel = medial_axis(keep_lab_img, return_distance=False)
ax3.imshow(skel)
ax3.set_title('Distance Map Ridge')


# # Morphological thinning
# From scikit-image documentation (http://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html)
# ```
# Morphological thinning, implemented in the thin function, works on the same principle as skeletonize: remove pixels from the borders at each iteration until none can be removed without altering the connectivity. The different rules of removal can speed up skeletonization and result in different final skeletons.
# 
# The thin function also takes an optional max_iter keyword argument to limit the number of thinning iterations, and thus produce a relatively thicker skeleton.``` 
# 
# We can use this to thin the tiny junk elements first then erode, then perform the full skeletonization

# In[9]:


from skimage.morphology import thin, erosion
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 7))
ax1.imshow(keep_lab_img)
ax1.set_title('Segmentation')

thin_image = thin(keep_lab_img, max_iter=1)
ax2.imshow(thin_image)
ax2.set_title('Morphologically Thinned')

er_thin_image = opening(thin_image, disk(1))
er_thin_image = label(er_thin_image) == 1
ax3.imshow(er_thin_image)
ax3.set_title('Opened')

opened_skeleton = medial_axis(er_thin_image, return_distance=False)
ax4.imshow(opened_skeleton)
ax4.set_title('Thinned/Opened Skeleton')


# ### Still overgrown
# The skeleton is still problematic for us and so we require some additional improvements to get a perfect skeleton

# # Skeleton: Junction Overview 
# 
# 
# With the skeleton which is ideally one voxel thick, we can characterize the junctions in the system by looking at the neighborhood of each voxel.
# 

# In[10]:


from scipy.ndimage import convolve

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
ax1.imshow(opened_skeleton)
ax1.set_title('Skeleton')
neighbor_conv = convolve(opened_skeleton.astype(int), np.ones((3, 3)))
neighbor_conv[~opened_skeleton] = 0
j_img = ax2.imshow(neighbor_conv,
                   cmap='nipy_spectral',
                   vmin=1, vmax=4,
                   interpolation='none')
plt.colorbar(j_img)
ax2.set_title('Junction Count')


# In[11]:


fig, ax1 = plt.subplots(1, 1, figsize=(8, 12))
n_crop = neighbor_conv[50:90, 40:60]
sns.heatmap(n_crop,
            annot=True,
            fmt="d",
            cmap='nipy_spectral',
            ax=ax1,
            cbar=True,
            vmin=0,
            vmax=n_crop.max(),
            annot_kws={"size": 10})


# In[12]:


junc_types = np.unique(neighbor_conv[neighbor_conv > 0])
fig, m_axs = plt.subplots(1, len(junc_types), figsize=(20, 7))
for i, c_ax in zip(junc_types, m_axs):
    c_ax.imshow(neighbor_conv == i, interpolation='none')
    c_ax.set_title('Count == {}'.format(i))


# ## Dedicated pruning algorithms
#  - Ideally model-based
#  - Minimum branch length (using component labeling on the Count==3)
#  - Minimum branch width (using the distance map values)
#  

# In[13]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
lab_seg = label(neighbor_conv == 3)
ax1.imshow(lab_seg, cmap='gist_earth')
ax1.set_title('Segment ID')
ax2.hist(lab_seg[lab_seg > 0])
ax2.set_title('Segment Length')

label_length_img = np.zeros_like(lab_seg)
for i in np.unique(lab_seg[lab_seg > 0]):
    label_length_img[lab_seg == i] = np.sum(lab_seg == i)

ll_ax = ax3.imshow(label_length_img, cmap='jet')
ax3.set_title('Segment Length')
plt.colorbar(ll_ax)


# In[14]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
length_skeleton = (label_length_img > 5) +     (neighbor_conv == 2)+(neighbor_conv > 3)
ax1.imshow(im_crop)
ax2.imshow(length_skeleton)
ax3.imshow(length_skeleton)
ax3.scatter(mk_crop['x'], mk_crop['y'], s=mk_crop['width'],
            alpha=0.25, label='Ground Truth')
ax3.legend()


# In[15]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

label_width_img = np.zeros_like(lab_seg)
for i in np.unique(lab_seg[lab_seg > 0]):
    label_width_img[lab_seg == i] = np.max(dist_map[lab_seg == i])

ax1.hist(label_width_img[label_width_img > 0])
ax1.set_title('Segment Maximum Width')


ll_ax = ax2.imshow(label_width_img, cmap='jet')
ax2.set_title('Segment Maximum Width')
plt.colorbar(ll_ax)


# In[16]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
width_skeleton = (label_width_img > 4.5) +     (neighbor_conv == 2)+(neighbor_conv > 3)
width_skeleton = label(width_skeleton) == 1
ax1.imshow(im_crop)
ax2.imshow(width_skeleton)
ax3.imshow(width_skeleton)
ax3.scatter(mk_crop['x'], mk_crop['y'], s=mk_crop['width'],
            alpha=0.25, label='Ground Truth')
ax3.legend()


# # Establish Topology
# From the cleaned, pruned skeleton we can start to establish topology. Using the same criteria as before we can break down the image into segments, junctions, and end-points

# In[17]:


ws_neighbors = convolve(width_skeleton.astype(
    int), np.ones((3, 3)), mode='constant', cval=0)
ws_neighbors[~width_skeleton] = 0
fig, (ax1) = plt.subplots(1, 1, figsize=(7, 12), dpi=150)
ax1.imshow(im_crop)
j_name = {1: 'dangling point', 2: 'end-point',
          3: 'segment', 4: 'junction', 5: 'super-junction'}
for j_count in np.unique(ws_neighbors[ws_neighbors > 0]):
    y_c, x_c = np.where(ws_neighbors == j_count)
    ax1.plot(x_c, y_c, 's',
             label=j_name.get(j_count, 'unknown'),
             markersize=8)

leg = ax1.legend(shadow=True, fancybox=True, frameon=True)


# # Getting Topology in Image Space
# We want to determine which nodes are directly connected in this image so we can extract a graph. If we take a simple case of two nodes connected by one edge and the bottom node connected to another edge going nowhere.
# 
# $$ \begin{bmatrix}
# n & 0 & 0 & 0 \\
# 0 & e & 0 & 0 \\
# 0 & 0 & n & e
# \end{bmatrix} $$
# 
# We can use component labeling to identify each node and each edge uniquely
# ## Node Labels
# 
# $$ N_{lab} = \begin{bmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 0 \\
# 0 & 0 & 2 & 0
# \end{bmatrix} $$
# 
# ## Edge Labels
# 
# $$E_{lab} = \begin{bmatrix}
# 0 & 0 & 0 & 0 \\
# 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & 2
# \end{bmatrix} $$
# 
# We can then use a dilation operation on the nodes and the edges to see which overlap

# In[18]:


from skimage.morphology import dilation
n_img = np.zeros((3, 4))
e_img = np.zeros_like(n_img)
n_img[0, 0] = 1
e_img[1, 1] = 1
n_img[2, 2] = 1
e_img[2, 3] = 1

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(8, 20))

ax1.imshow(n_img)
ax1.set_title('Nodes')

ax2.imshow(e_img)
ax2.set_title('Edges')

# labeling
n_labs = label(n_img)

sns.heatmap(n_labs, annot=True, fmt="d", ax=ax3, cbar=False)
ax3.set_title('Node Labels')

e_labs = label(e_img)

sns.heatmap(e_labs, annot=True, fmt="d", ax=ax4, cbar=False)
ax4.set_title('Edge Labels')

# growing
n_grow_1 = dilation(n_labs == 1, np.ones((3, 3)))

sns.heatmap(n_grow_1, annot=True, fmt="d", ax=ax5, cbar=False)
ax5.set_title('Grow First\n{} {}'.format('Edges Found:', [
              x for x in np.unique(e_labs[n_grow_1 > 0]) if x > 0]))

n_grow_2 = dilation(n_labs == 2, np.ones((3, 3)))
sns.heatmap(n_grow_2, annot=True, fmt="d", ax=ax6, cbar=False)
ax6.set_title('Grow Second\n{} {}'.format('Edges Found:', [
              x for x in np.unique(e_labs[n_grow_2 > 0]) if x > 0]))


# In[19]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
node_id_image = label((ws_neighbors > 3) | (ws_neighbors == 2))
edge_id_image = label(ws_neighbors == 3)

ax1.imshow(im_crop)

node_dict = {}
for c_node in np.unique(node_id_image[node_id_image > 0]):
    y_n, x_n = np.where(node_id_image == c_node)
    node_dict[c_node] = {'x': np.mean(x_n),
                         'y': np.mean(y_n),
                         'width': np.mean(dist_map[node_id_image == c_node])}
    ax1.plot(np.mean(x_n), np.mean(y_n), 'rs')

edge_dict = {}
edge_matrix = np.eye(len(node_dict)+1)
for c_edge in np.unique(edge_id_image[edge_id_image > 0]):
    edge_grow_mask = dilation(edge_id_image == c_edge, np.ones((3, 3)))
    v_nodes = np.unique(node_id_image[edge_grow_mask > 0])
    v_nodes = [v for v in v_nodes if v > 0]
    print('Edge', c_edge, 'connects', v_nodes)
    if len(v_nodes) == 2:
        edge_dict[c_edge] = {'start': v_nodes[0],
                             'end': v_nodes[-1],
                             'length': np.sum(edge_id_image == c_edge),
                             'euclidean_distance': np.sqrt(np.square(node_dict[v_nodes[0]]['x'] -
                                                                     node_dict[v_nodes[-1]]['x']) +
                                                           np.square(node_dict[v_nodes[0]]['y'] -
                                                                     node_dict[v_nodes[-1]]['y'])
                                                           ),
                             'max_width': np.max(dist_map[edge_id_image == c_edge]),
                             'mean_width': np.mean(dist_map[edge_id_image == c_edge])}
        edge_matrix[v_nodes[0], v_nodes[-1]] = np.sum(edge_id_image == c_edge)
        edge_matrix[v_nodes[-1], v_nodes[0]] = np.sum(edge_id_image == c_edge)
        s_node = node_dict[v_nodes[0]]
        e_node = node_dict[v_nodes[-1]]
        ax1.plot([s_node['x'], e_node['x']],
                 [s_node['y'], e_node['y']], 'b-', linewidth=np.mean(dist_map[edge_id_image == c_edge]), alpha=0.5)

ax2.matshow(edge_matrix, cmap='viridis')
ax2.set_title('Connectivity Matrix')


# # Skeleton: Tortuosity
# 
# One of the more interesting ones in material science is called tortuosity and it is defined as the ratio between the arc-length of a _segment_ and the distance between its starting and ending points. 
# $$ \tau = \frac{L}{C} $$
# 
# A high degree of [tortuosity](http://en.wikipedia.org/wiki/Tortuosity) indicates that the network is convoluted and is important when estimating or predicting flow rates.
# Specifically
# - in geology it is an indication that diffusion and fluid transport will occur more slowly
# - in analytical chemistry it is utilized to perform size exclusion chromatography
# - in vascular tissue it can be a sign of pathology.

# In[20]:


fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
ax1.imshow(im_crop)

for _, d_values in edge_dict.items():
    v_nodes = [d_values['start'], d_values['end']]
    print('Tortuousity: %2.2f' %
          (d_values['length']/d_values['euclidean_distance']))
    s_node = node_dict[v_nodes[0]]
    e_node = node_dict[v_nodes[-1]]
    ax1.plot([s_node['x'], e_node['x']],
             [s_node['y'], e_node['y']], 'b-',
             linewidth=5, alpha=d_values['length']/d_values['euclidean_distance'])


# In[21]:


import networkx as nx
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
G = nx.Graph()
for k, v in node_dict.items():
    G.add_node(k, weight=v['width'])
for k, v in edge_dict.items():
    G.add_edge(v['start'], v['end'], **v)
nx.draw_spring(G, ax=ax1, with_labels=True,
               node_color=[node_dict[k]['width']
                           for k in sorted(node_dict.keys())],
               node_size=800,
               cmap=plt.cm.autumn,
               edge_color=[G.edges[k]['length'] for k in list(G.edges.keys())],
               width=[2*G.edges[k]['max_width'] for k in list(G.edges.keys())],
               edge_cmap=plt.cm.Greens)
ax1.set_title('Randomly Organized Graph')
ax2.imshow(im_crop)
nx.draw(G,
        pos={k: (v['x'], v['y']) for k, v in node_dict.items()},
        ax=ax2,
        node_color=[node_dict[k]['width'] for k in sorted(node_dict.keys())],
        node_size=50,
        cmap=plt.cm.autumn,
        edge_color=[G.edges[k]['length'] for k in list(G.edges.keys())],
        width=[2*G.edges[k]['max_width'] for k in list(G.edges.keys())],
        edge_cmap=plt.cm.Blues,
        alpha=0.5,
        with_labels=False)


# ## Graph Analysis
# Once the data has been represented in a graph form, we can begin to analyze some of graph aspects of it, like the degree and connectivity plots.

# In[22]:


degree_sequence = sorted([d for n, d in G.degree()],
                         reverse=True)  # degree sequence
plt.hist(degree_sequence, bins=np.arange(10))


# # Watershed
# 
# 
# Watershed is a method for segmenting objects without using component labeling. 
# - It utilizes the shape of structures to find objects
# - From the distance map we can make out substructures with our eyes
# - But how to we find them?!
# 
# We use a sample image now from the [Datascience Bowl 2018 from Kaggle](https://www.kaggle.com/c/data-science-bowl-2018). The challenge is to identify nuclei in histology images to eventually find cancer better. The winner tweeted about the solution [here](https://twitter.com/alxndrkalinin/status/986260848376197120)
# 
# ![image.png](attachment:image.png)

# In[23]:


from skimage.filters import threshold_otsu
from skimage.color import rgb2hsv
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
get_ipython().run_line_magic('matplotlib', 'inline')

rgb_img = imread("../common/figures/dsb_sample/slide.png")[:, :, :3]
gt_labs = imread("../common/figures/dsb_sample/labels.png")
bw_img = rgb2hsv(rgb_img)[:, :, 2]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6), dpi=100)
ax1.imshow(rgb_img, cmap='bone')
ax2.imshow(bw_img, cmap='bone')
ax2.set_title('Gray Scale')
ax3.imshow(bw_img < threshold_otsu(bw_img), cmap='bone')
ax3.set_title('Segmentation')
ax4.imshow(gt_labs, cmap='gist_earth')


# In[24]:


from skimage.morphology import label
import seaborn as sns
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6), dpi=100)
bw_roi = bw_img[75:110:2, 125:150:2]
ax1.imshow(bw_roi, cmap='bone')
ax1.set_title('Gray Scale')
bw_roi_seg = bw_roi < threshold_otsu(bw_img)
sns.heatmap(bw_roi_seg, annot=True, fmt="d",
            ax=ax2, cbar=False, cmap='gist_earth')
ax2.set_title('Segmentation')
bw_roi_label = label(bw_roi_seg)
sns.heatmap(bw_roi_label, annot=True, fmt="d",
            ax=ax3, cbar=False, cmap='gist_earth')
ax3.set_title('Labels')
sns.heatmap(gt_labs[75:110:2, 125:150:2], annot=True,
            fmt="d", ax=ax4, cbar=False, cmap='gist_earth')
ax4.set_title('Ground Truth')


# # Watershed: Flowing Downhill
# 
# 
# We can imagine watershed as waterflowing down hill into basins. The topology in this case is given by the distance map

# In[25]:


from scipy.ndimage import distance_transform_edt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(15, 10))
ax = fig.gca(projection='3d')
bw_roi_dmap = distance_transform_edt(bw_roi_seg)

# Plot the surface.
t_xx, t_yy = np.meshgrid(np.arange(bw_roi_dmap.shape[1]),
                         np.arange(bw_roi_dmap.shape[0]))
surf = ax.plot_surface(t_xx, t_yy,
                       -1*bw_roi_dmap,
                       cmap=plt.cm.viridis_r,
                       linewidth=0.1,
                       antialiased=True)

# Customize the z axis.
ax.view_init(60, 20)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5)


# In[26]:


from skimage.feature import peak_local_max
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
sns.heatmap(bw_roi_seg, annot=True, fmt="d",
            ax=ax1, cbar=False, cmap='gist_earth')
ax1.set_title('Segmentation')
sns.heatmap(bw_roi_dmap, annot=True, fmt="1.0f",
            ax=ax2, cbar=False, cmap='viridis')
ax2.set_title('Distance Map')
roi_local_maxi = peak_local_max(bw_roi_dmap, indices=False, footprint=np.ones((3, 3)),
                                labels=bw_roi_seg, exclude_border=False)
labeled_maxi = label(roi_local_maxi)

sns.heatmap(labeled_maxi, annot=True, fmt="1.0f",
            ax=ax3, cbar=False, cmap='gist_earth')
ax3.set_title('Local Maxima')


# In[27]:


from skimage.morphology import watershed
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
sns.heatmap(labeled_maxi, annot=True, fmt="1.0f",
            ax=ax1, cbar=False, cmap='gist_earth')
ax1.set_title('Local Maxima')

ws_labels = watershed(-bw_roi_dmap, labeled_maxi, mask=bw_roi_seg)

sns.heatmap(ws_labels, annot=True, fmt="d",
            ax=ax2, cbar=False, cmap='gist_earth')
ax2.set_title('Watershed')

sns.heatmap(gt_labs[75:110:2, 125:150:2], annot=True,
            fmt="d", ax=ax3, cbar=False, cmap='gist_earth')
ax3.set_title('Ground Truth')


# # Removing too small elements
# We see here that one of the comonents (2) is too small. We can remove it by deleting objects in the bottom 10 percentile of areas and then rerunning watershed

# In[28]:


label_area_dict = {i: np.sum(ws_labels == i)
                   for i in np.unique(ws_labels[ws_labels > 0])}
clean_label_maxi = labeled_maxi.copy()
area_cutoff = np.percentile(list(label_area_dict.values()), 10)
print('!0% cutoff', area_cutoff)
for i, k in label_area_dict.items():
    print('Label: ', i, 'Area:', k, 'Keep:', k > area_cutoff)
    if k <= area_cutoff:
        clean_label_maxi[clean_label_maxi == i] = 0


# In[29]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
sns.heatmap(clean_label_maxi, annot=True, fmt="1.0f",
            ax=ax1, cbar=False, cmap='gist_earth')
ax1.set_title('Local Maxima')

ws_labels = watershed(-bw_roi_dmap, clean_label_maxi, mask=bw_roi_seg)

sns.heatmap(ws_labels, annot=True, fmt="d",
            ax=ax2, cbar=False, cmap='gist_earth')
ax2.set_title('Watershed')

sns.heatmap(gt_labs[75:110:2, 125:150:2], annot=True,
            fmt="d", ax=ax3, cbar=False, cmap='gist_earth')
ax3.set_title('Ground Truth')


# # Scaling back up
# Now we can perform the operation on the whole image and see how the results look
# 

# In[30]:


from skimage.morphology import opening, disk
from skimage.segmentation import mark_boundaries
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
ax1.imshow(rgb_img, cmap='bone')

bw_seg_img = opening(bw_img < threshold_otsu(bw_img), disk(3))

bw_dmap = distance_transform_edt(bw_seg_img)
bw_peaks = label(peak_local_max(bw_dmap, indices=False, footprint=np.ones((3, 3)),
                                labels=bw_seg_img, exclude_border=True))

ws_labels = watershed(-bw_dmap, bw_peaks, mask=bw_seg_img)

label_area_dict = {i: np.sum(ws_labels == i)
                   for i in np.unique(ws_labels[ws_labels > 0])}

clean_label_maxi = bw_peaks.copy()
lab_areas = list(label_area_dict.values())
area_cutoff = np.percentile(lab_areas, 20)
print('10% cutoff', area_cutoff, 'Removed', np.sum(
    np.array(lab_areas) < area_cutoff), 'components')
for i, k in label_area_dict.items():
    if k <= area_cutoff:
        clean_label_maxi[clean_label_maxi == i] = 0

ws_labels = watershed(-bw_dmap, clean_label_maxi, mask=bw_seg_img)

ax2.imshow(mark_boundaries(label_img=ws_labels,
                           image=rgb_img, color=(0, 1, 0)))
ax2.set_title('Watershed')

ax3.imshow(mark_boundaries(label_img=gt_labs, image=rgb_img, color=(0, 1, 0)))
ax3.set_title('Ground Truth')


# # Battery Example
# 
# We use an example from the Laboratory for Nanoelectronics at ETH Zurich. The datasets are x-ray tomography images of the battery micro- and nanostructures. As the papers below document, a substantial amount of image processing is required to extract meaningful physical and chemical values from these images.
# 
# ## Acknowledgements
# The relevant publications which also contain links to the full collection of datasets.
# 
# X‐Ray Tomography of Porous, Transition Metal Oxide Based Lithium Ion Battery Electrodes - https://onlinelibrary.wiley.com/doi/full/10.1002/aenm.201200932
# 
# Quantifying Inhomogeneity of Lithium Ion Battery Electrodes and Its Influence on Electrochemical Performance - http://jes.ecsdl.org/content/165/2/A339.abstract

# ## Goal
# THe goal is to segment and quantify the relevant structures to find out what changes occur between 0 and 2000 bar of pressure. 

# In[31]:


from scipy.ndimage import binary_fill_holes
from skimage.morphology import opening, closing, disk
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
get_ipython().run_line_magic('matplotlib', 'inline')
bw_img = imread("../common/data/NMC_90wt_2000bar_115.tif")[:, :, 0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10), dpi=120)
ax1.imshow(bw_img, cmap='bone')
ax1.set_title('Gray Scale')
thresh_img = bw_img > threshold_otsu(bw_img)
ax2.imshow(thresh_img, cmap='bone')
ax2.set_title('Segmentation')
bw_seg_img = closing(
    closing(
        opening(thresh_img, disk(3)),
        disk(1)
    ), disk(1)
)
#bw_seg_img = binary_fill_holes(bw_seg_img)
ax3.imshow(bw_seg_img, cmap='bone')
ax3.set_title('Clean Segments')


# In[32]:


from scipy.ndimage import distance_transform_edt
from skimage.morphology import label
from skimage.feature import peak_local_max
from skimage.segmentation import mark_boundaries

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,  36), dpi=100)

bw_dmap = distance_transform_edt(bw_seg_img)

ax1.imshow(bw_dmap, cmap='nipy_spectral')

bw_peaks = label(peak_local_max(bw_dmap, indices=False, footprint=np.ones((3, 3)),
                                labels=bw_seg_img, exclude_border=True))

ws_labels = watershed(-bw_dmap, bw_peaks, mask=bw_seg_img)

ax2.imshow(ws_labels, cmap='gist_earth')
# find boundaries
ax3.imshow(mark_boundaries(label_img=ws_labels, image=bw_img))
ax3.set_title('Boundaries')


# # Representing as a Graph
# Here we can change the representation from a number of random labels to a graph

# In[33]:


from skimage.morphology import dilation
from skimage.measure import perimeter
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))


def get_roi(x): return x[0:200, 200:450]


im_crop = get_roi(bw_img)
dist_map = get_roi(bw_dmap)
node_id_image = get_roi(ws_labels)

ax1.imshow(im_crop)

node_dict = {}
for c_node in np.unique(node_id_image[node_id_image > 0]):
    y_n, x_n = np.where(node_id_image == c_node)
    node_dict[c_node] = {'x': np.mean(x_n),
                         'y': np.mean(y_n),
                         'width': np.mean(dist_map[node_id_image == c_node]),
                         'perimeter': perimeter(node_id_image == c_node)}
    ax1.plot(np.mean(x_n), np.mean(y_n), 'rs')

edge_dict = {}

for i in node_dict.keys():
    i_grow = dilation(node_id_image == i, np.ones((3, 3)))
    for j in node_dict.keys():
        if i < j:
            j_grow = dilation(node_id_image == j, np.ones((3, 3)))
            interface_length = np.sum(i_grow & j_grow)
            if interface_length > 0:
                v_nodes = [i, j]

                edge_dict[(i, j)] = {'start': v_nodes[0],
                                     'start_perimeter': node_dict[v_nodes[0]]['perimeter'],
                                     'end_perimeter': node_dict[v_nodes[-1]]['perimeter'],
                                     'end': v_nodes[-1],
                                     'interface_length': interface_length,
                                     'euclidean_distance': np.sqrt(np.square(node_dict[v_nodes[0]]['x'] -
                                                                             node_dict[v_nodes[-1]]['x']) +
                                                                   np.square(node_dict[v_nodes[0]]['y'] -
                                                                             node_dict[v_nodes[-1]]['y'])
                                                                   ),
                                     'max_width': np.max(dist_map[i_grow & j_grow]),
                                     'mean_width': np.mean(dist_map[i_grow & j_grow])}
                s_node = node_dict[v_nodes[0]]
                e_node = node_dict[v_nodes[-1]]
                ax1.plot([s_node['x'], e_node['x']],
                         [s_node['y'], e_node['y']], 'b-',
                         linewidth=np.max(dist_map[i_grow & j_grow]), alpha=0.5)

ax2.imshow(mark_boundaries(label_img=node_id_image, image=im_crop))
ax2.set_title('Borders')


# In[34]:


import pandas as pd
edge_df = pd.DataFrame(list(edge_dict.values()))
edge_df.head(5)


# # Combine split electrodes
# Here we combine split electrodes by using a cutoff on the ratio of the interface length to the start and end perimeters

# In[35]:


delete_edges = edge_df.query(
    'interface_length>0.33*(start_perimeter+end_perimeter)')
print('Found', delete_edges.shape[0], '/', edge_df.shape[0], 'edges to delete')
delete_edges.head(5)


# In[36]:


node_id_image = get_roi(ws_labels)
for _ in range(3):
    # since some mappings might be multistep
    for _, c_row in delete_edges.iterrows():
        node_id_image[node_id_image == c_row['end']] = c_row['start']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

ax1.imshow(im_crop)

node_dict = {}
for c_node in np.unique(node_id_image[node_id_image > 0]):
    y_n, x_n = np.where(node_id_image == c_node)
    node_dict[c_node] = {'x': np.mean(x_n),
                         'y': np.mean(y_n),
                         'width': np.mean(dist_map[node_id_image == c_node]),
                         'perimeter': perimeter(node_id_image == c_node)}
    ax1.plot(np.mean(x_n), np.mean(y_n), 'rs')

edge_dict = {}

for i in node_dict.keys():
    i_grow = dilation(node_id_image == i, np.ones((3, 3)))
    for j in node_dict.keys():
        if i < j:
            j_grow = dilation(node_id_image == j, np.ones((3, 3)))
            interface_length = np.sum(i_grow & j_grow)
            if interface_length > 0:
                v_nodes = [i, j]

                edge_dict[(i, j)] = {'start': v_nodes[0],
                                     'start_perimeter': node_dict[v_nodes[0]]['perimeter'],
                                     'end_perimeter': node_dict[v_nodes[-1]]['perimeter'],
                                     'end': v_nodes[-1],
                                     'interface_length': interface_length,
                                     'euclidean_distance': np.sqrt(np.square(node_dict[v_nodes[0]]['x'] -
                                                                             node_dict[v_nodes[-1]]['x']) +
                                                                   np.square(node_dict[v_nodes[0]]['y'] -
                                                                             node_dict[v_nodes[-1]]['y'])
                                                                   ),
                                     'max_width': np.max(dist_map[i_grow & j_grow]),
                                     'mean_width': np.mean(dist_map[i_grow & j_grow])}
                s_node = node_dict[v_nodes[0]]
                e_node = node_dict[v_nodes[-1]]
                ax1.plot([s_node['x'], e_node['x']],
                         [s_node['y'], e_node['y']], 'b-',
                         linewidth=np.max(dist_map[i_grow & j_grow]), alpha=0.5)

ax2.imshow(mark_boundaries(label_img=node_id_image, image=im_crop))
ax2.set_title('Borders')


# In[37]:


import networkx as nx
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
G = nx.Graph()
for k, v in node_dict.items():
    G.add_node(k, weight=v['width'])
for k, v in edge_dict.items():
    G.add_edge(v['start'], v['end'], **v)
nx.draw_shell(G, ax=ax1, with_labels=True,
              node_color=[node_dict[k]['width']
                          for k in sorted(node_dict.keys())],
              node_size=400,
              cmap=plt.cm.autumn,
              edge_color=[G.edges[k]['interface_length']
                          for k in list(G.edges.keys())],
              width=[2*G.edges[k]['max_width'] for k in list(G.edges.keys())],
              edge_cmap=plt.cm.Greens)
ax1.set_title('Randomly Organized Graph')
ax2.imshow(im_crop)
nx.draw(G,
        pos={k: (v['x'], v['y']) for k, v in node_dict.items()},
        ax=ax2,
        node_color=[node_dict[k]['width'] for k in sorted(node_dict.keys())],
        node_size=50,
        cmap=plt.cm.Greens,
        edge_color=[G.edges[k]['interface_length']
                    for k in list(G.edges.keys())],
        width=[2*G.edges[k]['max_width'] for k in list(G.edges.keys())],
        edge_cmap=plt.cm.autumn,
        alpha=0.75,
        with_labels=False)


# In[38]:


degree_sequence = sorted([d for n, d in G.degree()],
                         reverse=True)  # degree sequence
plt.hist(degree_sequence, bins=np.arange(10))


# In[ ]:




