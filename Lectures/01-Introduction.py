#!/usr/bin/env python
# coding: utf-8

#
# # ETHZ: 227-0966-00L
# # Quantitative Big Imaging
# # February 21, 2019
#
# ## Introduction and Overview
#

# ## Overview
#
# - Who are we?
# - Who are you?
#  - What is expected?
# - __Why does this class exist?__
#  - Collection
#  - Changing computing (Parallel / Cloud)
#  - Course outline

# ## Overview
#
# - What is an image?
# - Where do images come from?
# - Science and Reproducibility
# - Workflows

# # Who are we?
#
#
# ## Kevin Mader (mader@biomed.ee.ethz.ch)
#  - CTO at __4Quant__ for Big Image Analytics (ETH Spin-off)
#  - __Lecturer__ at ETH Zurich
#  - Formerly __Postdoc__ in the X-Ray Microscopy Group at ETH Zurich (2013-2015)
#  - PhD Student at Swiss Light Source at Paul Scherrer Institute (2008-2012)
#
# - ![Kevin Mader](https://avatars0.githubusercontent.com/u/116120?s=460&v=4)

# ## Amogha Pandeshwar (Amogha.Pandeshwar@psi.ch)
#  - Exercise assistance
#  - __PhD Student__ in the X-Ray Microscopy Group at ETH Zurich and Swiss Light Source at Paul Scherrer Institute
#
# - ![Amogha](https://www.psi.ch/lsb/PeopleUebersichtsTabelle/APandeshwar_small.jpg)

# # Guest Lecturers
#
# ## Anders Kaestner, PhD (anders.kaestner@psi.ch)
#  - __Group Leader__ at the ICON Beamline at the SINQ (Neutron Source) at Paul Scherrer Institute
#
# ![Anders Kaestner](../common/figures/anders.png)

# ## Michael Prummer, PhD (prummer@nexus.ethz.ch)
#
# - Biostatistician at NEXUS Personalized Health Technol.
# - Previously Senior Scientist at F. Hoffmann-La Roche Ltd., Basel, Switzerland.
#  - Pharma Research & Early Development (pRED), Discovery Technologies
#  - Phenotypic Drug Discovery & Target Identification.
#  - Topic: High Content Screening (HCS), Image analysis, Biostatistics, Image Management System.
#
# ![Michael Prummer](http://www.nexus.ethz.ch/content/specialinterest/dual/nexus/en/people/person-detail.person_image.jpeg?persid=90104)

# ## Who are you?
#
#
# ### A wide spectrum of backgrounds
# - Biomedical Engineers, Physicists, Chemists, Art History Researchers, Mechanical Engineers, and Computer Scientists
#
# ### A wide range of skills
# - I think I've heard of Matlab before $\rightarrow$ I write template C++ code and hand optimize it afterwards

# # So how will this ever work?
#
# ## Adaptive assignments
#
# ### Conceptual, graphical assignments with practical examples
#   - Emphasis on chosing correct steps and understanding workflow
#
# ### Opportunities to create custom implementations, plugins, and perform more complicated analysis on larger datasets if interested
#   - Emphasis on performance, customizing analysis, and scalability

# # Course Expectations
#
# ## Exercises
#  - Usually 1 set per lecture
#  - Optional (but recommended!)
#  - Easy - using GUIs (KNIME and ImageJ) and completing Matlab Scripts (just lecture 2)
#  - Advanced - Writing Python, Java, Scala, ...
#
# ## Science Project
# - Optional (but strongly recommended)
# - Applying Techniques to answer scientific question!
#  - Ideally use on a topic relevant for your current project, thesis, or personal activities
#  - or choose from one of ours (will be online, soon)
# - Present approach, analysis, and results

# # Literature / Useful References
#
#
# ## General Material
# - Jean Claude, Morphometry with R
#  - [Online](http://link.springer.com/book/10.1007%2F978-0-387-77789-4) through ETHZ
#  - [Buy it](http://www.amazon.com/Morphometrics-R-Use-Julien-Claude/dp/038777789X)
# - John C. Russ, “The Image Processing Handbook”,(Boca Raton, CRC Press)
#  - Available [online](http://dx.doi.org/10.1201/9780203881095) within domain ethz.ch (or proxy.ethz.ch / public VPN)
# - J. Weickert, Visualization and Processing of Tensor Fields
#  - [Online](http://books.google.ch/books?id=ScLxPORMob4C&lpg=PA220&ots=mYIeQbaVXP&dq=&pg=PA220#v=onepage&q&f=false)

# ## Today's Material
#
#
# - Imaging
#  - [ImageJ and SciJava](http://www.slideshare.net/CurtisRueden/imagej-and-the-scijava-software-stack)
# - Cloud Computing
#  - [The Case for Energy-Proportional Computing](http://www-inst.eecs.berkeley.edu/~cs61c/sp14/) _ Luiz André Barroso, Urs Hölzle, IEEE Computer, December 2007_
#  - [Concurrency](www.gotw.ca/publications/concurrency-ddj.htm)
# - Reproducibility
#  - [Trouble at the lab](http://www.economist.com/news/briefing/21588057-scientists-think-science-self-correcting-alarming-degree-it-not-trouble) _Scientists like to think of science as self-correcting. To an alarming degree, it is not_
#  - [Why is reproducible research important?](http://simplystatistics.org/2014/06/06/the-real-reason-reproducible-research-is-important/) _The Real Reason Reproducible Research is Important_
#  - [Science Code Manifesto](http://software-carpentry.org/blog/2011/10/the-science-code-manifestos-five-cs.html)
#  - [Reproducible Research Class](https://www.coursera.org/course/repdata) @ Johns Hopkins University

# # Motivation
#
# ![Crazy Workflow](../common/figures/crazyworkflow.png)
# - To understand what, why and how from the moment an image is produced until it is finished (published, used in a report, …)
# - To learn how to go from one analysis on one image to 10, 100, or 1000 images (without working 10, 100, or 1000X harder)

# - Detectors are getting bigger and faster constantly
# - Todays detectors are really fast
#  - 2560 x 2160 images @ 1500+ times a second = 8GB/s
# - Matlab / Avizo / Python / … are saturated after 60 seconds
# - A single camera
#  - [More information per day than Facebook](http://news.cnet.com/8301-1023_3-57498531-93/facebook-processes-more-than-500-tb-of-data-daily/)
#  - [Three times as many images per second as Instagram](http://techcrunch.com/2013/01/17/instagram-reports-90m-monthly-active-users-40m-photos-per-day-and-8500-likes-per-second/)

# ### X-Ray
#  - SRXTM images at (>1000fps) → 8GB/s
#  - cSAXS diffraction patterns at 30GB/s
#  - Nanoscopium Beamline, 10TB/day, 10-500GB file sizes
#
# ### Optical
#  - Light-sheet microscopy (see talk of Jeremy Freeman) produces images → 500MB/s
#  - High-speed confocal images at (>200fps) → 78Mb/s
#
# ### Personal
#  - GoPro 4 Black - 60MB/s (3840 x 2160 x 30fps) for $600
#  - [fps1000](https://www.kickstarter.com/projects/1623255426/fps1000-the-low-cost-high-frame-rate-camera) - 400MB/s (640 x 480 x 840 fps) for $400

# ## Motivation
#
#
# 1. __Experimental Design__ finding the right technique, picking the right dyes and samples has stayed relatively consistent, better techniques lead to more demanding scientits.
#
# 2. __Management__ storing, backing up, setting up databases, these processes have become easier and more automated as data magnitudes have increased
#
# 3. __Measurements__ the actual acquisition speed of the data has increased wildly due to better detectors, parallel measurement, and new higher intensity sources
#
# 4. __Post Processing__ this portion has is the most time-consuming and difficult and has seen minimal improvements over the last years
#
# ----

# ![Experiment Breakdown](../common/figures/experiment-breakdown.png)

#
# ## How much is a TB, really?
#
#
# If __you__ looked at one 1000 x 1000 sized image
#

# In[43]:


import matplotlib.pyplot as plt
import numpy as np

plt.matshow(np.random.uniform(size=(1000, 1000)), cmap="viridis")


# every second, it would take you
#

# In[2]:


# assuming 16 bit images and a 'metric' terabyte
time_per_tb = 1e12 / (1000 * 1000 * 16 / 8) / (60 * 60)
print("%04.1f hours to view a terabyte" % (time_per_tb))


# ## Overwhelmed
#
# - Count how many cells are in the bone slice
# - Ignore the ones that are ‘too big’ or shaped ‘strangely’
# - Are there more on the right side or left side?
# - Are the ones on the right or left bigger, top or bottom?
#
#
# ![cells in bone tissue](../common/figures/bone-cells.png)

# ## More overwhelmed
#
# - Do it all over again for 96 more samples, this time with 2000 slices instead of just one!
#
#
# ![more samples](../common/figures/96-samples.png)

# ## Bring on the pain
#
# - Now again with 1090 samples!
#
#
# ![even more samples](../common/figures/1090-samples.png)

# ## It gets better
#
#
# - Those metrics were quantitative and could be easily visually extracted from the images
# - What happens if you have _softer_ metrics
#
#
# ![alignment](../common/figures/alignment-figure.png)
#
#
# - How aligned are these cells?
# - Is the group on the left more or less aligned than the right?
# - errr?

# ## Dynamic Information
#
# <video controls>
#   <source src="../common/movies/dk31-plat.avi" type="video/avi">
# Your browser does not support the video tag.
# </video>
#
#
#
# - How many bubbles are here?
# - How fast are they moving?
# - Do they all move the same speed?
# - Do bigger bubbles move faster?
# - Do bubbles near the edge move slower?
# - Are they rearranging?

# # Computing has changed: Parallel
#
#
# ## Moores Law
# $$ \textrm{Transistors} \propto 2^{T/(\textrm{18 months})} $$

# In[3]:


# stolen from https://gist.github.com/humberto-ortiz/de4b3a621602b78bf90d
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

moores_txt = [
    "Id Name  Year  Count(1000s)  Clock(MHz)\n",
    "0            MOS65XX  1975           3.51           14\n",
    "1          Intel8086  1978          29.00           10\n",
    "2          MIPSR3000  1988         120.00           33\n",
    "3           AMDAm486  1993        1200.00           40\n",
    "4        NexGenNx586  1994        3500.00          111\n",
    "5          AMDAthlon  1999       37000.00         1400\n",
    "6   IntelPentiumIII  1999       44000.00         1400\n",
    "7         PowerPC970  2002       58000.00         2500\n",
    "8       AMDAthlon64  2003      243000.00         2800\n",
    "9    IntelCore2Duo  2006      410000.00         3330\n",
    "10         AMDPhenom  2007      450000.00         2600\n",
    "11      IntelCorei7  2008     1170000.00         3460\n",
    "12      IntelCorei5  2009      995000.00         3600",
]

sio_table = StringIO("".join(moores_txt))
moore_df = pd.read_table(sio_table, sep="\s+", index_col=0)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
ax1.semilogy(
    moore_df["Year"], moore_df["Count(1000s)"], "b.-", label="1000s of transitiors"
)
ax1.semilogy(moore_df["Year"], moore_df["Clock(MHz)"], "r.-", label="Clockspeed (MHz)")
ax1.legend(loc=2)


# <small>_Based on data from https://gist.github.com/humberto-ortiz/de4b3a621602b78bf90d_</small>
#
# ----
#
# There are now many more transistors inside a single computer but the processing speed hasn't increased. How can this be?
#
# - Multiple Core
#  - Many machines have multiple cores for each processor which can perform tasks independently
# - Multiple CPUs
#  - More than one chip is commonly present
# - New modalities
#   - GPUs provide many cores which operate at slow speed
#
# ### Parallel Code is important

# ## Computing has changed: Cloud

# - Computer, servers, workstations are wildly underused (majority are <50%)
# - Buying a big computer that sits idle most of the time is a waste of money
#
# <small>http://www-inst.eecs.berkeley.edu/~cs61c/sp14/
# “The Case for Energy-Proportional Computing,” Luiz André Barroso, Urs Hölzle, IEEE Computer, December 2007</small>
#
# ![cloud services](../common/figures/cloud-services.png)

# - Traditionally the most important performance criteria was time, how fast can it be done
# - With Platform as a service servers can be rented instead of bought
# - Speed is still important but using cloud computing $ / Sample is the real metric
# - In Switzerland a PhD student if 400x as expensive per hour as an Amazon EC2 Machine
# - Many competitors keep prices low and offer flexibility

# ## Cloud Computing Costs
#
#
# The figure shows the range of cloud costs (determined by peak usage) compared to a local workstation with utilization shown as the average number of hours the computer is used each week.

#
# ## Cloud: Equal Cost Point
#
# Here the equal cost point is shown where the cloud and local workstations have the same cost. The x-axis is the percentage of resources used at peak-time and the y shows the expected usable lifetime of the computer. The color indicates the utilization percentage and the text on the squares shows this as the numbers of hours used in a week.
#
#
#

#
# # Course Overview
#

# In[8]:


import json, pandas as pd

course_df = pd.read_json("../common/schedule.json")
course_df["Date"] = course_df["Lecture"].map(lambda x: x.split("-")[0])
course_df["Title"] = course_df["Lecture"].map(lambda x: x.split("-")[-1])
course_df[["Date", "Title", "Description"]]


#
# ## Overview: Segmentation
#
#

# In[9]:


course_df[["Title", "Description", "Applications"]][3:6].T


#
# ## Overview: Analysis

# In[11]:


course_df[["Title", "Description", "Applications"]][6:9].T


#
# ## Overview: Big Imaging
#
#

# In[13]:


course_df[["Title", "Description", "Applications"]][9:12].T


#
# ## Overview: Wrapping Up
#
#

# In[16]:


course_df[["Title", "Description", "Applications"]][12:13].T


#
# # What is an image?
#
# ----
#
# A very abstract definition: __A pairing between spatial information (position) and some other kind of information (value).__
#
# In most cases this is a 2 dimensional position (x,y coordinates) and a numeric value (intensity)
#
#

# In[9]:


basic_image = np.random.choice(range(100), size=(5, 5))
xx, yy = np.meshgrid(range(basic_image.shape[1]), range(basic_image.shape[0]))
image_df = pd.DataFrame(dict(x=xx.ravel(), y=yy.ravel(), Intensity=basic_image.ravel()))
image_df[["x", "y", "Intensity"]].head(5)


# In[10]:


plt.matshow(basic_image, cmap="viridis")
plt.colorbar()


#
# ## 2D Intensity Images
#
# The next step is to apply a color map (also called lookup table, LUT) to the image so it is a bit more exciting
#
#

# In[46]:


fig, ax1 = plt.subplots(1, 1)
plot_image = ax1.matshow(basic_image, cmap="Blues")
plt.colorbar(plot_image)

for _, c_row in image_df.iterrows():
    ax1.text(
        c_row["x"], c_row["y"], s="%02d" % c_row["Intensity"], fontdict=dict(color="r")
    )


# Which can be arbitrarily defined based on how we would like to visualize the information in the image

# In[12]:


fig, ax1 = plt.subplots(1, 1)
plot_image = ax1.matshow(basic_image, cmap="jet")
plt.colorbar(plot_image)


# In[13]:


fig, ax1 = plt.subplots(1, 1)

plot_image = ax1.matshow(basic_image, cmap="hot")
plt.colorbar(plot_image)


#
# ## Lookup Tables
#
# Formally a lookup table is a function which
# $$ f(\textrm{Intensity}) \rightarrow \textrm{Color} $$
#
#
#

# In[14]:


import matplotlib.pyplot as plt
import numpy as np

xlin = np.linspace(0, 1, 100)
fig, ax1 = plt.subplots(1, 1)
ax1.scatter(xlin, plt.cm.hot(xlin)[:, 0], c=plt.cm.hot(xlin))
ax1.scatter(xlin, plt.cm.Blues(xlin)[:, 0], c=plt.cm.Blues(xlin))

ax1.scatter(xlin, plt.cm.jet(xlin)[:, 0], c=plt.cm.jet(xlin))

ax1.set_xlabel("Intensity")
ax1.set_ylabel("Red Component")


#
#
# These transformations can also be non-linear as is the case of the graph below where the mapping between the intensity and the color is a $\log$ relationship meaning the the difference between the lower values is much clearer than the higher ones
#
#
# ## Applied LUTs
#

# In[15]:


import matplotlib.pyplot as plt
import numpy as np

xlin = np.logspace(-2, 5, 500)
log_xlin = np.log10(xlin)
norm_xlin = (log_xlin - log_xlin.min()) / (log_xlin.max() - log_xlin.min())
fig, ax1 = plt.subplots(1, 1)

ax1.scatter(xlin, plt.cm.hot(norm_xlin)[:, 0], c=plt.cm.hot(norm_xlin))

ax1.scatter(xlin, plt.cm.hot(xlin / xlin.max())[:, 0], c=plt.cm.hot(norm_xlin))
ax1.set_xscale("log")
ax1.set_xlabel("Intensity")
ax1.set_ylabel("Red Component")


#
# On a real image the difference is even clearer
#

# In[16]:


import matplotlib.pyplot as plt
from skimage.io import imread

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
in_img = imread("../common/figures/bone-section.png")[:, :, 0].astype(np.float32)
ax1.imshow(in_img, cmap="gray")
ax1.set_title("grayscale LUT")

ax2.imshow(in_img, cmap="hot")
ax2.set_title("hot LUT")

ax3.imshow(np.log2(in_img + 1), cmap="gray")
ax3.set_title("grayscale-log LUT")


#
#
# ## 3D Images
#
# For a 3D image, the position or spatial component has a 3rd dimension (z if it is a spatial, or t if it is a movie)
#
#

# In[17]:


import numpy as np

vol_image = np.arange(27).reshape((3, 3, 3))
print(vol_image)


#
#
#
# This can then be rearranged from a table form into an array form and displayed as a series of slices
#
#

# In[18]:


import matplotlib.pyplot as plt
from skimage.util import montage as montage2d

print(montage2d(vol_image, fill=0))
plt.matshow(montage2d(vol_image, fill=0), cmap="jet")


# ## Multiple Values
#
# In the images thus far, we have had one value per position, but there is no reason there cannot be multiple values. In fact this is what color images are (red, green, and blue) values and even 4 channels with transparency (alpha) as a different. For clarity we call the __dimensionality__ of the image the number of dimensions in the spatial position, and the __depth__ the number in the value.
#
#

# In[19]:


import pandas as pd
from itertools import product
import numpy as np

base_df = pd.DataFrame([dict(x=x, y=y) for x, y in product(range(5), range(5))])
base_df["Intensity"] = np.random.uniform(0, 1, 25)
base_df["Transparency"] = np.random.uniform(0, 1, 25)
base_df.head(5)


#
# This can then be rearranged from a table form into an array form and displayed as a series of slices
#

# In[20]:


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(base_df["x"], base_df["y"], c=plt.cm.gray(base_df["Intensity"]), s=1000)
ax1.set_title("Intensity")
ax2.scatter(base_df["x"], base_df["y"], c=plt.cm.gray(base_df["Transparency"]), s=1000)
ax2.set_title("Transparency")


# In[21]:


fig, (ax1) = plt.subplots(1, 1)
ax1.scatter(
    base_df["x"],
    base_df["y"],
    c=plt.cm.jet(base_df["Intensity"]),
    s=1000 * base_df["Transparency"],
)
ax1.set_title("Intensity")


# ## Hyperspectral Imaging
#
#
# At each point in the image (black dot), instead of having just a single value, there is an entire spectrum. A selected group of these (red dots) are shown to illustrate the variations inside the sample. While certainly much more complicated, this still constitutes and image and requires the same sort of techniques to process correctly.
#
#

# In[22]:


import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
import os

raw_img = imread(os.path.join("..", "common", "data", "raw.jpg"))
im_pos = pd.read_csv(os.path.join("..", "common", "data", "impos.csv"), header=None)
im_pos.columns = ["x", "y"]
fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
ax1.imshow(raw_img)
ax1.scatter(im_pos["x"], im_pos["y"], s=1, c="blue")


# In[23]:


full_df = pd.read_csv(os.path.join("..", "common", "data", "full_img.csv")).query(
    "wavenum<1200"
)
print(full_df.shape[0], "rows")
full_df.head(5)


# In[24]:


full_df["g_x"] = pd.cut(full_df["x"], 5)
full_df["g_y"] = pd.cut(full_df["y"], 5)
fig, m_axs = plt.subplots(5, 5, figsize=(12, 12))
for ((g_x, g_y), c_rows), c_ax in zip(
    full_df.sort_values(["x", "y"]).groupby(["g_x", "g_y"]), m_axs.flatten()
):
    c_ax.plot(c_rows["wavenum"], c_rows["val"], "r.")


# # Image Formation
#
#
# ![Traditional Imaging](../common/figures/image-formation.png)
#
# - __Impulses__ Light, X-Rays, Electrons, A sharp point, Magnetic field, Sound wave
# - __Characteristics__ Electron Shell Levels, Electron Density, Phonons energy levels, Electronic, Spins, Molecular mobility
# - __Response__ Absorption, Reflection, Phase Shift, Scattering, Emission
# - __Detection__ Your eye, Light sensitive film, CCD / CMOS, Scintillator, Transducer

# ## Where do images come from?

# In[25]:


import pandas as pd
from io import StringIO

pd.read_table(
    StringIO(
        """Modality\tImpulse	Characteristic	Response	Detection
Light Microscopy	White Light	Electronic interactions	Absorption	Film, Camera
Phase Contrast	Coherent light	Electron Density (Index of Refraction)	Phase Shift	Phase stepping, holography, Zernike
Confocal Microscopy	Laser Light	Electronic Transition in Fluorescence Molecule	Absorption and reemission	Pinhole in focal plane, scanning detection
X-Ray Radiography	X-Ray light	Photo effect and Compton scattering	Absorption and scattering	Scintillator, microscope, camera
Ultrasound	High frequency sound waves	Molecular mobility	Reflection and Scattering	Transducer
MRI	Radio-frequency EM	Unmatched Hydrogen spins	Absorption and reemission	RF coils to detect
Atomic Force Microscopy	Sharp Point	Surface Contact	Contact, Repulsion	Deflection of a tiny mirror"""
    )
)


#
# # Acquiring Images
#
# ## Traditional / Direct imaging
# - Visible images produced or can be easily made visible
# - Optical imaging, microscopy
#

# In[26]:


import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from scipy.ndimage import convolve
from skimage.morphology import disk
import numpy as np
import os

bone_img = imread(os.path.join("..", "common", "figures", "tiny-bone.png")).astype(
    np.float32
)
# simulate measured image
conv_kern = np.pad(disk(2), 1, "constant", constant_values=0)
meas_img = convolve(bone_img[::-1], conv_kern)
# run deconvolution
dekern = np.fft.ifft2(1 / np.fft.fft2(conv_kern))
rec_img = convolve(meas_img, dekern)[::-1]
# show result
fig, (ax_orig, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))
ax_orig.imshow(bone_img, cmap="bone")
ax_orig.set_title("Original Object")

ax1.imshow(meas_img, cmap="bone")
ax1.set_title("Measurement")

ax2.imshow(rec_img, cmap="bone", vmin=0, vmax=255)
ax2.set_title("Reconstructed")


# ## Indirect / Computational imaging
# - Recorded information does not resemble object
# - Response must be transformed (usually computationally) to produce an image
#
#

# In[27]:


import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from scipy.ndimage import convolve
from skimage.morphology import disk
import numpy as np
import os

bone_img = imread(os.path.join("..", "common", "figures", "tiny-bone.png")).astype(
    np.float32
)
# simulate measured image
meas_img = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(bone_img))))
print(meas_img.min(), meas_img.max(), meas_img.mean())
fig, (ax1, ax_orig) = plt.subplots(1, 2, figsize=(12, 6))
ax_orig.imshow(bone_img, cmap="bone")
ax_orig.set_title("Original Object")

ax1.imshow(meas_img, cmap="hot")
ax1.set_title("Measurement")


# ## Traditional Imaging
#
#
# ![Traditional Imaging](../common/figures/traditional-imaging.png)
#
#
# <small>
# Copyright 2003-2013 J. Konrad in EC520 lecture, reused with permission
# </small>

# ## Traditional Imaging: Model
#
#
# ![Traditional Imaging Model](../common/figures/traditional-image-flow.png)
#
# $$
# \left[\left([b(x,y)*s_{ab}(x,y)]\otimes h_{fs}(x,y)\right)*h_{op}(x,y)\right]*h_{det}(x,y)+d_{dark}(x,y)
# $$
#
# $s_{ab}$ is the only information you are really interested in, so it is important to remove or correct for the other components
#
# For color (non-monochromatic) images the problem becomes even more complicated
# $$
# \int_{0}^{\infty} {\left[\left([b(x,y,\lambda)*s_{ab}(x,y,\lambda)]\otimes h_{fs}(x,y,\lambda)\right)*h_{op}(x,y,\lambda)\right]*h_{det}(x,y,\lambda)}\mathrm{d}\lambda+d_{dark}(x,y)
# $$

# ## Indirect Imaging (Computational Imaging)
#
# - Tomography through projections
# - Microlenses (Light-field photography)
#
# <video controls>
#   <source src="../common/movies/lightfield.mp4" type="video/mp4">
# Your browser does not support the video tag.
# </video>
#
#
# - Diffraction patterns
# - Hyperspectral imaging with Raman, IR, CARS
# - Surface Topography with cantilevers (AFM)
#
# ![Suface Topography](../common/figures/surface-plot.png)

# ## Image Analysis
#
#
# ![Approaches](../common/figures/approaches.png)
#
#
# - An image is a bucket of pixels.
# - How you choose to turn it into useful information is strongly dependent on your background

# ## Image Analysis: Experimentalist
#
#
# ![Approaches](../common/figures/approaches.png)
#
#
# ### Problem-driven
# ### Top-down
# ### _Reality_ Model-based
#
# ### Examples
#
# - cell counting
# - porosity

# ## Image Analysis: Computer Vision Approaches
#
#
# ![Approaches](../common/figures/approaches.png)
#
#
# - Method-driven
#  - Feature-based
#  - _Image_ Model-based
# - Engineer features for solving problems
#
# ### Examples
#
# - edge detection
# - face detection
#

# ## Image Analysis: Deep Learning Approach
#
#
# ![Approaches](../common/figures/approaches.png)
#
#
# - Results-driven
# - Biology ‘inspired’
# - Build both image processing and analysis from scratch
#
# ### Examples
#
# - Captioning images
# - Identifying unusual events
#

# # On Science
#
# ## What is the purpose?
#
#
# - Discover and validate new knowledge
#
# ### How?
# - Use the scientific method as an approach to convince other people
# - Build on the results of others so we don't start from the beginning
#
# ### Important Points
# - While qualitative assessment is important, it is difficult to reliably produce and scale
#  - __Quantitative__ analysis is far from perfect, but provides metrics which can be compared and regenerated by anyone
#
# <small>Inspired by: [imagej-pres](http://www.slideshare.net/CurtisRueden/imagej-and-the-scijava-software-stack)</small>

# ## Science and Imaging
#
# ### Images are great for qualitative analyses since our brains can quickly interpret them without large _programming_ investements.
# ### Proper processing and quantitative analysis is however much more difficult with images.
#  - If you measure a temperature, quantitative analysis is easy, $50K$.
#  - If you measure an image it is much more difficult and much more prone to mistakes, subtle setup variations, and confusing analyses
#
#
# ### Furthermore in image processing there is a plethora of tools available
#
# - Thousands of algorithms available
# - Thousands of tools
# - Many images require multi-step processing
# - Experimenting is time-consuming

# ## Why quantitative?
#
# ### Human eyes have issues
#
# Which center square seems brighter?

# In[28]:


import matplotlib.pyplot as plt
import numpy as np

xlin = np.linspace(-1, 1, 3)
xx, yy = np.meshgrid(xlin, xlin)
img_a = 25 * np.ones((3, 3))
img_b = np.ones((3, 3)) * 75
img_a[1, 1] = 50
img_b[1, 1] = 50
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
ax1.matshow(img_a, vmin=0, vmax=100, cmap="bone")
ax2.matshow(img_b, vmin=0, vmax=100, cmap="bone")


#
# ----
# Are the intensities constant in the image?
#
#

# In[29]:


import matplotlib.pyplot as plt
import numpy as np

xlin = np.linspace(-1, 1, 10)
xx, yy = np.meshgrid(xlin, xlin)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
ax1.matshow(xx, vmin=-1, vmax=1, cmap="bone")


# ## Reproducibility
#
# Science demands __repeatability__! and really wants __reproducability__
# - Experimental conditions can change rapidly and are difficult to make consistent
# - Animal and human studies are prohibitively time consuming and expensive to reproduce
# - Terabyte datasets cannot be easily passed around many different groups
# - Privacy concerns can also limit sharing and access to data
#
# ----
#
# - _Science_ is already difficult enough
# - Image processing makes it even more complicated
# - Many image processing tasks are multistep, have many parameters, use a variety of tools, and consume a very long time
#
# ### How can we keep track of everything for ourselves and others?
# - We can make the data analysis easy to repeat by an independent 3rd party

# # Soup/Recipe Example
#
# ## Simple Soup
# Easy to follow the list, anyone with the right steps can execute and repeat (if not reproduce) the soup
#
#
# 1. Buy {carrots, peas, tomatoes} at market
# 1. _then_ Buy meat at butcher
# 1. _then_ Chop carrots into pieces
# 1. _then_ Chop potatos into pieces
# 1. _then_ Heat water
# 1. _then_ Wait until boiling then add chopped vegetables
# 1. _then_ Wait 5 minutes and add meat
#
#
#
# ## More complicated soup
# Here it is harder to follow and you need to carefully keep track of what is being performed
#
# ### Steps 1-4
# 4. _then_ Mix carrots with potatos $\rightarrow  mix_1$
# 4. _then_ add egg to $mix_1$ and fry for 20 minutes
# 4. _then_ Tenderize meat for 20 minutes
# 4. _then_ add tomatoes to meat and cook for 10 minutes $\rightarrow mix_2$
# 5. _then_ Wait until boiling then add $mix_1$
# 6. _then_ Wait 5 minutes and add $mix_2$

# # Using flow charts / workflows
#
# ## Simple Soup

# In[1]:


from IPython.display import SVG
import pydot

graph = pydot.Dot(graph_type="digraph")
node_names = [
    "Buy\nvegetables",
    "Buy meat",
    "Chop\nvegetables",
    "Heat water",
    "Add Vegetables",
    "Wait for\nboiling",
    "Wait 5\nadd meat",
]
nodes = [pydot.Node(name="%04d" % i, label=c_n) for i, c_n in enumerate(node_names)]
for c_n in nodes:
    graph.add_node(c_n)

for (c_n, d_n) in zip(nodes, nodes[1:]):
    graph.add_edge(pydot.Edge(c_n, d_n))

SVG(graph.create_svg())


#
# ## Workflows
#
# Clearly a linear set of instructions is ill-suited for even a fairly easy soup, it is then even more difficult when there are dozens of steps and different pathsways
#
#
# ----
#
# Furthermore a clean workflow allows you to better parallelize the task since it is clear which tasks can be performed independently
#
#

# In[2]:


from IPython.display import SVG
import pydot

graph = pydot.Dot(graph_type="digraph")
node_names = [
    "Buy\nvegetables",
    "Buy meat",
    "Chop\nvegetables",
    "Heat water",
    "Add Vegetables",
    "Wait for\nboiling",
    "Wait 5\nadd meat",
]
nodes = [
    pydot.Node(name="%04d" % i, label=c_n, style="filled")
    for i, c_n in enumerate(node_names)
]
for c_n in nodes:
    graph.add_node(c_n)


def e(i, j, col=None):
    if col is not None:
        for c in [i, j]:
            if nodes[c].get_fillcolor() is None:
                nodes[c].set_fillcolor(col)
    graph.add_edge(pydot.Edge(nodes[i], nodes[j]))


e(0, 2, "red")
e(2, 4)
e(3, -2, "yellow")
e(-2, 4, "orange")
e(4, -1)
e(1, -1, "green")


SVG(graph.create_svg())


#
# # Directed Acyclical Graphs (DAG)
# We can represent almost any computation without loops as DAG. What this allows us to do is now break down a computation into pieces which can be carried out independently. There are a number of tools which let us handle this issue.
#
# - PyData Dask - https://dask.pydata.org/en/latest/
# - Apache Spark - https://spark.apache.org/
# - Spotify Luigi - https://github.com/spotify/luigi
# - Airflow - https://airflow.apache.org/
# - KNIME - https://www.knime.com/
# - Google Tensorflow - https://www.tensorflow.org/
# - Pytorch / Torch - http://pytorch.org/

# # Concrete example
# What is a DAG good for?

# In[32]:


import dask.array as da
from dask.dot import dot_graph

image_1 = da.zeros((5, 5), chunks=(5, 5))
image_2 = da.ones((5, 5), chunks=(5, 5))
dot_graph(image_1.dask)


# In[33]:


image_3 = image_1 + image_2
dot_graph(image_3.dask)


# In[34]:


image_4 = (image_1 - 10) + (image_2 * 50)
dot_graph(image_4.dask)


# # Let's go big
# Now let's see where this can be really useful

# In[35]:


import dask.array as da
from dask.dot import dot_graph

image_1 = da.zeros((1024, 1024), chunks=(512, 512))
image_2 = da.ones((1024, 1024), chunks=(512, 512))
dot_graph(image_1.dask)


# In[36]:


image_4 = (image_1 - 10) + (image_2 * 50)
dot_graph(image_4.dask)


# In[37]:


image_5 = da.matmul(image_1, image_2)
dot_graph(image_5.dask)


# In[38]:


image_6 = (da.matmul(image_1, image_2) + image_1) * image_2
dot_graph(image_6.dask)


# In[39]:


import dask_ndfilters as da_ndfilt

image_7 = da_ndfilt.convolve(image_6, image_1)
dot_graph(image_7.dask)


# # Deep Learning
# We won't talk too much about deep learning now, but it certainly shows why DAGs are so important. The steps above are simple toys compared to what tools are already in use for machine learning

# In[3]:


from IPython.display import SVG
from keras.applications.resnet50 import ResNet50
from keras.utils.vis_utils import model_to_dot

resnet = ResNet50(weights=None)
SVG(model_to_dot(resnet).create_svg())


# In[5]:


from IPython.display import clear_output, Image, display, HTML
import keras.backend as K
import tensorflow as tf
import numpy as np


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == "Const":
            tensor = n.attr["value"].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = ("<stripped %d bytes>" % size).encode("ascii")
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, "as_graph_def"):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(
        data=repr(str(strip_def)), id="graph" + str(np.random.rand())
    )

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(
        code.replace('"', "&quot;")
    )
    display(HTML(iframe))


sess = K.get_session()
show_graph(sess.graph)


# In[ ]:
