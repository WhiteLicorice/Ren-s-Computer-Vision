# 😎 Ren's Computer Vision 😎

This is a repository for lab work in CMSC 174 (Computer Vision) with OpenCV-Python.

##  Lab 01

Create a Python script that replicates the process in this [video](https://fb.watch/pWLNqOIQPE/) using OpenCV-Python.

##  Lab 02

Create an image filtering function and use it to create hybrid images using a simplified version of the SIGGRAPH 2006 paper by Oliva, Torralba, and Schyns. Hybrid images are static images that change in interpretation as a function of the viewing distance. The basic idea is that high frequency tends to dominate perception when it is available, but, at a distance, only the low frequency (smooth) part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances. Use your own solution to create your own hybrid images. The provided file hybrid.py contains functions that you need to implement.

For just this laboratory exercise, you are forbidden from using any Numpy, Scipy, OpenCV, or other preimplemented functions for filtering. You are allowed to use basic matrix operations like np.shape, np.zeros, and np.transpose. This limitation will be lifted in future laboratory exercises , but for now, you should use for loops or Numpy vectorization to apply a kernel to each pixel in the image. The bulk of your code will be in cross_correlation, and gaussian_blur with the other functions using these functions either directly or through one of the other functions you implement.

Your pair of images needs to be aligned using an image manipulation software, e.g., Photoshop, Gimp.  Alignments can map the eyes to eyes and nose to nose, edges to edges, etc. It is encouraged to create additional examples (e.g. change of expression, morph between different objects, change over time, etc.). See the [hybrid images project page](http://olivalab.mit.edu/hybrid_gallery/gallery.html) for some inspiration. The project page also contains materials from their Siggraph presentation.

##  Lab 03

The goal of this laboratory exercise is to blend two images seamlessly using a multi resolution blending as described in the 1983 [paper](https://persci.mit.edu/pub_pdfs/spline83.pdf) by Burt and Adelson. An image spline is a smooth seam joining two image together by gently distorting them. Multiresolution blending computes a gentle seam between the two images seperately at each band of image frequencies, resulting in a much smoother seam. You are to create and visualize the Gaussian and Laplacian stacks of the input images, blending together images with the help of the completed stacks, and explore creative outcomes. If you would rather work with pyramids, you may implement pyramids rather than stacks. However, in any case, you are not allowed to use existing pyramid (pyrDown, pyrUp) functions. You must implement your stacks/pyramids from scratch.

The results are as follows.

<table>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/81cb1413-c3d0-4bed-a5fd-3dbb56db40dc" alt="apple"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/99daf091-a219-4f17-b9f2-b6f4227f004e" alt="orange"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/b77c6dd2-6c22-425d-a52b-016d5db6262b" alt="mask"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/2ed8fea1-9076-4a8a-8403-55899c640d8f" alt="blended apple orange"></td>
  </tr>
  <caption>Figure 3.1: Blending an Apple and an Orange with a Vertical Mask (Orapple)</caption>
</table>

<table>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/5324e65a-a370-4998-80e4-bbfd7e5b98e1" alt="pawn"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/3b7405b6-b492-46ad-b0c4-875adcec6ed2" alt="bishop"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/1339c0ed-5456-4b3c-a846-d62ac6461588" alt="crazymask"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/54efd071-1eef-44dd-88df-39b2f4958181" alt="crazy"></td>
  </tr>
  <caption>Figure 3.2: Blending Two Chessboards with an Irregular Mask (Not Your Typical Endgame)</caption>
</table>
