# 😎 Ren's Computer Vision 😎

This is a repository for lab work in CMSC 174 (Computer Vision) with OpenCV-Python.

##  Lab 01

Create a Python script that replicates the process in this [video](https://fb.watch/pWLNqOIQPE/) using OpenCV-Python.

<table>
  <caption><i>Figure 1.1: Playing Around With Perspective</i></caption>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/db6927d7-e9fc-4f41-a06b-6e1b8164de51" alt="Output"></td
  </tr>
</table>


##  Lab 02

Create an image filtering function and use it to create hybrid images using a simplified version of the SIGGRAPH 2006 paper by Oliva, Torralba, and Schyns. Hybrid images are static images that change in interpretation as a function of the viewing distance. The basic idea is that high frequency tends to dominate perception when it is available, but, at a distance, only the low frequency (smooth) part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances. Use your own solution to create your own hybrid images. The provided file hybrid.py contains functions that you need to implement.

For just this laboratory exercise, you are forbidden from using any Numpy, Scipy, OpenCV, or other preimplemented functions for filtering. You are allowed to use basic matrix operations like np.shape, np.zeros, and np.transpose. This limitation will be lifted in future laboratory exercises , but for now, you should use for loops or Numpy vectorization to apply a kernel to each pixel in the image. The bulk of your code will be in cross_correlation, and gaussian_blur with the other functions using these functions either directly or through one of the other functions you implement.

Your pair of images needs to be aligned using an image manipulation software, e.g., Photoshop, Gimp.  Alignments can map the eyes to eyes and nose to nose, edges to edges, etc. It is encouraged to create additional examples (e.g. change of expression, morph between different objects, change over time, etc.). See the [hybrid images project page](http://olivalab.mit.edu/hybrid_gallery/gallery.html) for some inspiration. The project page also contains materials from their Siggraph presentation.

<table>
  <caption><i>Figure 2.1: Hondarrari Construction</i></caption>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/d09ea735-4388-43a4-a90e-88af767fa714" alt="Honda" width="200" height="200"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/7609ee6b-44df-48f2-a4cf-f38fca0ab494" alt="Ferrari" width="200" height="200"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/ca2c0930-f93d-4a62-9eff-c292934f1a14" alt="Hondarrari" width="200" height="200"></td>
  </tr>
</table>



##  Lab 03

The goal of this laboratory exercise is to blend two images seamlessly using a multi resolution blending as described in the 1983 [paper](https://persci.mit.edu/pub_pdfs/spline83.pdf) by Burt and Adelson. An image spline is a smooth seam joining two image together by gently distorting them. Multiresolution blending computes a gentle seam between the two images seperately at each band of image frequencies, resulting in a much smoother seam. You are to create and visualize the Gaussian and Laplacian stacks of the input images, blending together images with the help of the completed stacks, and explore creative outcomes. If you would rather work with pyramids, you may implement pyramids rather than stacks. However, in any case, you are not allowed to use existing pyramid (pyrDown, pyrUp) functions. You must implement your stacks/pyramids from scratch.

The results are as follows.

<table>
  <caption><i>Figure 3.1: Blending an Apple and an Orange with a Vertical Mask (Orapple)</i></caption>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/81cb1413-c3d0-4bed-a5fd-3dbb56db40dc" alt="apple" width="200" height="200"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/99daf091-a219-4f17-b9f2-b6f4227f004e" alt="orange" width="200" height="200"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/b77c6dd2-6c22-425d-a52b-016d5db6262b" alt="mask" width="200" height="200"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/1de1860f-8191-4009-8983-05c6e4238fc2" alt="blended apple orange" width="200" height="200"></td>
  </tr>
</table>

<table>
  <caption><i>Figure 3.2: Blending Two Chessboards with an Irregular Mask (Not Your Typical Endgame)</i></caption>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/5324e65a-a370-4998-80e4-bbfd7e5b98e1" alt="pawn" width="200" height="200"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/3b7405b6-b492-46ad-b0c4-875adcec6ed2" alt="bishop" width="200" height="200"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/1339c0ed-5456-4b3c-a846-d62ac6461588" alt="crazymask" width="200" height="200"></td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/3fbb9406-bac8-449a-9d0a-2dc89146426e" alt="crazy" width="200" height="200"></td>
  </tr>
</table>


##  Lab 04

The goal of this laboratory exercise is to estimate the amount of liquid contained in a bottle. Given images of a bottle with unknown amounts of liquid, you are to devise a method for guessing these amounts. OpenCV image filtering, thresholding, or morphology operations are allowed.

<table>
  <caption><b>Known Amounts</b></caption>
  <tr>
    <th>Amount</th>
    <th>Bottle</th>
  </tr>
  <tr>
    <td>50ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/b110fd6d-0dd6-45c2-b971-5cec287a31a5" alt="50ml" width="200" height="200"></td>
  </tr>
  <tr>
    <td>100ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/8d794511-caa8-4a85-88de-75bfb33ed546" alt="100ml" width="200" height="200"></td>
  </tr>
  <tr>
    <td>150ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/eb66141e-0ba7-42f3-be7b-7319c2ce81c3" alt="150ml" width="200" height="200"></td>
  </tr>
  <tr>
    <td>200ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/f0a9e8e5-78b0-49a0-ac5b-9cf2a027dbb5" alt="200ml" width="200" height="200"></td>
  </tr>
  <tr>
    <td>250ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/838fed9b-27af-4345-b8ce-3535036089df" alt="250ml" width="200" height="200"></td>
  </tr>
  <tr>
    <td>300ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/748efc69-6b4d-4bd0-b789-abd15cc3bc4c" alt="300ml" width="200" height="200"></td>
  </tr>
  <tr>
    <td>350ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/40dedec7-e7ea-4a68-9874-73c4c5f55971" alt="350ml" width="200" height="200"></td>
  </tr>
</table>

<table>
  <caption><b>Predicted Amounts</b></caption>
  <tr>
    <th>Amount</th>
    <th>Bottle</th>
  </tr>
  <tr>
    <td>122ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/ec4313fb-e587-4afc-9850-63cd601fe019" alt="122ml" width="200" height="200"></td>
  </tr>
  <tr>
    <td>207ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/89b96486-30ec-42fa-8b9b-2e5a1ed270a2" alt="207ml" width="200" height="200"></td>
  </tr>
  <tr>
    <td>345ml</td>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/65d15a2d-04a7-4442-9c2e-ec53e5c2b8ef" alt="345ml" width="200" height="200"></td>
  </tr>
</table>

##  Lab 05
With this [article](https://medium.com/@paulsonpremsingh7/image-stitching-using-opencv-a-step-by-step-tutorial-9214aa4255ec) as basis, stitch the provided images of a spread from a book. Once you have devised a method for stitching images, acquire a video and generate an actionshot image.  [Actionshot](https://en.wikipedia.org/wiki/ActionShot) is a method of capturing an object in action and displaying it in a single image with multiple sequential appearances of the object. SAFETY FIRST. Be sure you know what you are doing should you choose to record your own video.

<table>
  <caption><i>Figure 5.1: An Odd Book Spread</i></caption>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/b08f712b-06b0-4722-b97a-324a612ba1fd" alt="stitched_image"></td>
  </tr>
</table>

<table>
  <caption><i>Figure 5.2: Spike Action Shot</i></caption>
  <tr>
    <td align="center"><img src="https://github.com/WhiteLicorice/Ren-s-Computer-Vision/assets/96515086/1d51b89f-678d-42fa-9bff-26ca7e420299" alt="action_shot"></td>
  </tr>
</table>


