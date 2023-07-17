Note inspired [khanhha](https://khanhha.github.io) articles:
- [Thin Plate Splines Warping](https://khanhha.github.io/posts/Thin-Plate-Splines-Warping/)
- [Image-Based Virtual Try On Network - Part 1](https://khanhha.github.io/posts/Image-based-Virtual-Try-On-Network-Part-1/)



# Problem
Given an image with a sparse set of control points $(x_i,y_i)$ with corresponding displacements $(\Delta x_i,\Delta y_i)$ , we want to find a mapping $f:(x,y)→(x′,y′)$ from pixels in the input image to pixels in the warped/deformed image so that the corresponding warped control points $(x′_i,y′_i)$ closely match its expected targets $(x_i+\Delta x_i,y_i + \Delta y_i)$, and the surrounding points are deformed as smoothly as possible.

![warping_chess](https://user-images.githubusercontent.com/51177049/230336535-958a24e0-1d12-4def-b830-1ca8ecdd54c7.png)

In the image above, we want to warp the origin point of each red arrow with the final point of each arrow.



# U-Net Module

### Calculate person representation: $p$
The input person is image is transformed to another person representation to get rid of information about old clothes, color, texture and shape and still preserves face, hair and general body shape of the target.
Such representation consists of:
- A pose heat map → 18-channel image where each slice encodes the heat map of a skeleton joint.
- A body shape map → 1-channel image which describes the blurred shape of the person.
- An RGB image that contains the facial and hair region of the human subject.


### Find feature correlation
After the person representation p is extracted from the input image, with the cloth image c, they are passed to two separate feature extraction modules, each of which consists of a chain of 2-strided down-sampling convolutional, batch normalization and relu layers.
Both extracted features are then passed to a correlation module that is supposed to merge them into a single tensor that encodes the correlation between the person pose and the standard in-shop clothes.


### Predict Thin-Plate-Spline transformation
The correlation map is then passed to a regressor (the blue trapeze block) that predicts the warped control points for the Thin-Plate-Spline stage.
These blue control points will be then used to solve for a smooth Thin-Plate-Spline transformation that warps the input in-shop clothes images to align with the target clothes images on the human subject.
In other words, the Thin-Plate-Spline transformation is learned by minimizing the MSE loss between the warped clothes and the corresponding target clothes.
![warped_cloth](https://user-images.githubusercontent.com/51177049/230336605-2da8df1f-5545-41f1-9170-5e70753b2307.png)

Given two sets of grid control points, the module **TpsGridGen** estimates a Thin Plate Spline transformation that warps the in-shop clothes to match the person pose.
**The first control point set**, as shown in the left picture, is constructed in the initialization stage and does not change during the training process. **The second control point set** in the right picture is the prediction result from the regressor.
![tps_grd_gen](https://user-images.githubusercontent.com/51177049/230336683-24069613-202c-4784-a2e6-6b660ed067ce.png)

The estimated TPS transformation will be then used to sample a dense pixel mapping, which maps the pixels in the in-shop clothes image to the pixels in the domain of the target clothes image so that the final warped clothes image aligns with the human subject.
