<!-- Problem Statement -->

The [Diabetic Retinopathy challenge on Kaggle][KL] has just finished. The goal of the
competition was to predict the presence and severity of the disease [Diabetic Retinopathy](https://en.wikipedia.org/wiki/Diabetic_retinopathy) from photographs of eyes. I finished in [20th place][KL] using a Convolutional Neural Network (ConvNet). In this post I'll explain my learning process and progress as I implemented my first ConvNet over the last 3 months. Throughout, I'll link to the implementations in my code, which is [available on github](https://github.com/ilyakava/kaggle-dr) for anyone who wishes to replicate my score.

![**My progress over all my 170+ experiments**. See the Misc section at the end for a list of the improvements that each point represents here as written along the x-axis. *Each point represents an experiment that set a personal record high (on a validation set) of the Kappa score that the competition is judged by. Each point contains a description of the change that caused the improvement (each improvement is accumulated over the experiments, i.e. later runs include all the past improvements).*](http://f.cl.ly/items/0i1M3h310Y0n411L0D40/summary2.png)

# Introduction

## The Problem

Diabetic Retinopathy (DR) is one of the most significant complications of diabetes and is a leading cause of blindness. Early detection and treatment is essential for preventing blindness. Ophthalmologists can use a [lens](https://en.wikipedia.org/wiki/Ophthalmoscopy) to look through the dilated pupils of a patient and see the retina at the back of the eyeball, looking for symptoms that indicate changes in blood vessels ([NIH][NIH]). At worst this means that new blood vessels are growing (proliferative DR or PDR) and disturbing the retina, otherwise the patient has non-proliferative DR (NPDR). For the challenge, there are 5 stages of DR severity that have symptoms (from [here](http://www.icoph.org/downloads/Diabetic-Retinopathy-Scale.pdf)):

0. non-pathological, no NPDR
1. Mild NPDR, microanuerisms (red blotches) which are the source of hard exudate (high contrast yellow spots) sometimes in circulate patterns
2. Moderate NPDR. "More than just microaneurysms," perhaps cotton wool spots (fuzzy light blotches)
3. Severe NPDR: IRMA (shunt vessels), venous bleeding in 2+ quadrants, 20+ intra-retinal hemorrhages, no signs PDR
4. Neovascularization (often vessels with loops or very squiggly vessels), vitreous/preretinal hemorrhage, PDR

![**Examples of pathological classes 1,2,3,4 from left to right**](http://f.cl.ly/items/3s0239201r2A000l3Y3W/pathological.png)

The goal of the competition is to build a classifier that takes in these images, and outputs an integer diagnosis 0-4.

## My Approach: Convnets

![**A ConvNet correctly predicts class 2** *This plot shows the original 128x128 training image (41188_right) with four heatmaps. Each heatmap corresponds to one of four pathological classes. Each pixel in the heatmap represents the probability that that class is true given that pixel is corrupted in the original image. The heatmaps were created by moving around a 11x11 pixel block which hid a region of the image at testing time. The low probability areas (blue) mean that without those pixels that class would not be predicted. Here the network has learned to find the subtle hard exudate and (less so) cotton wool spots indicate classes 1 and 2 (since those classes would not be likely if those areas were obfuscated).*](http://f.cl.ly/items/0s3D1G2J402n2y1s0F0P/modelX_41188_right.png)

Since my last [Machine Learning class](http://www.cs.columbia.edu/~jebara/4772/), I've been looking forward to using ConvNets because of the promise of end-to-end learning: learning a feature extractor and a classifier simultaneously. This property allows accurate classifiers to be created without much domain knowledge. So in May, I read through a [Stanford tutorial][karpathy] and [Columbia reading list](http://llcao.net/cu-deeplearning15/reading.html) and toured [theano](http://deeplearning.net/software/theano/) while [implementing the best performing ConvNet on MNIST](http://github.com/ilyakava/ciresan). After that, I moved onto the more challenging DR dataset, with guidance from Sander Dieleman's posts on his two Kaggle ConvNet wins: classifying [galaxies][galaxy] and [plankton][plankton].

## Software and Hardware

I used same software setup as [Sander Dieleman in his Galaxy post][galaxy]. For details see my [repo][GH]. For hardware, I had access to a Nvidia M2090 with 6Gb RAM, and at the last moment a K40. Because of these assumptions, I use the `lasagne.layers.cuda_convnet` module which requires a GPU to run.

Of the networks I include, [128x128 runs](https://github.com/ilyakava/kaggle-dr/blob/master/network_specs.json#L183) (kappa ~0.68) took me 3.5 mins/epoch, [192x192 runs](https://github.com/ilyakava/kaggle-dr/blob/master/network_specs.json#L282) (kappa ~0.72) took 7.8 mins/epoch, and [256x256 runs](https://github.com/ilyakava/kaggle-dr/blob/master/network_specs.json#L282) (kappa ~0.74) took 14 mins/epoch.

# Preprocessing

## [Resizing with Graphicsmagick](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/create_resize_batchfiles.py#L29)

The supplied data consisted of JPEGs which were often 16 megapixels. For every size that I experimented with (128,152,192,236,256,292) my downsampling was the same strategy: crop out the surrounding black, size down the width, then vertically crop/letterbox until the image is square. I used PNGs because they are my favorite lossless format.

I played around with the idea of removing black pixels by using a [log-polar transformation](https://ganymed.imib.rwth-aachen.de/irma/ps-pdf/paper_fundus.pdf) on the images, but was unhappy with the distortion effect since the images are not all aligned in the same way (macula is not always centered).

## [Input = Raw Pixels](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/data_stream.py#L217)

Other than subtracting the mean image, and dividing by the standard deviation image, I made no modifications to the images entering the network. Early runs used grayscale images to lessen runtime.

### [Normalization](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/create_resize_batchfiles.py#L48) had no gains

For input into the network, histogram normalization with [graphicsmagick](http://www.graphicsmagick.org/GraphicsMagick.html#details-normalize) gave marginal improvements using grayscale images, and slightly worse performance when using color images. I never tried local contrast normalization simply because it has fallen out of favor.

## Exploiting Invariances by adding Noise

### [Flipping](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/data_stream.py#L15)

I randomly flipped/didn't flip an image along the horizontal axis, and then again for the vertical axis every time it entered the network. I liked that this transformation didn't change the image quality or the appearance of the letterbox which was generally symmetric. I didn't try rotating the eyeball a random degree because of the potential extra delay when loading the image into the network (and also because I feared the rotated letterbox might add harmful noise to the dataside), but given more time I would have tried this.

### [Color Casting](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/data_stream.py#L69)

I first heard about this from a [paper out of Baidu](http://arxiv.org/abs/1501.02876). I randomly decided on each channel whether or not to add/subtract a constant, and then drew this constant from a centered gaussian distribution of a standard deviation 10. This means that 99% of the values I added were between [-30,30]. I experimented with other ranges, ±30 worked best for me.

## Reducing Noise

### [Alignment](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/align_util.py#L147)

Early on, I had the belief that any reduction in noise in the training set would improve my convergence. For this reason, I decided to align all the training set images. Some images display a tab jutting out of the eyeball in the right half of the image. When this is the case, this means the image is [inverted](http://www.olympusmicro.com/primer/images/magnification/convexlens3.jpg). I wrote a pattern recognition style [tab detector](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/align_util.py#L114) that with 90% accuracy would detect this tab, and with the additional information of right/left eye from the image name, would output the flip that would put the optic nerve in the right of the image.

![A sample of 115 images after alignment, most of the time the optic nerve is on the right.](http://f.cl.ly/items/2Z1t251X0a2D1Y442w3s/unnamed.gif)

This pre-alignment gave a 1% improvement over the same run with the images oriented as they are found in the training set. However random flipping provided a 10% benefit, so this work did not prove useful.

### [Noise decay](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L387)

After some experiments I started to believe that for ConvNets, adding the right kind of noise prevents overfitting by roughening the error surface of the network, and lowering the energy barriers to hop out of bad local minima. I thought that by having high noise in the beginning of training, and less noise at the end, I would have the advantage of getting out of bad local minima early on, and being able to stay in good minima later on in training.

So I tried decaying the amount of noise: randomly flipping pre-aligned images less often as training progressed. I was surprised to find that shortly after every time I reduced the noise, my network began to overfit. I rationalized this as me reducing the amount of data (total uniq images) that the net had access to over time.

# [Network Architecture](https://github.com/ilyakava/kaggle-dr/blob/master/network_specs.json#L183)

The basis for my network is the popular [VGGNet from Oxford](http://arxiv.org/abs/1409.1556) that [Andrej Karpathy](http://cs231n.github.io/convolutional-networks/) recommends.

I tried several variations, all of which are in my [`network_specs.json`](https://github.com/ilyakava/kaggle-dr/blob/master/network_specs.json), but the [most successful one](https://github.com/ilyakava/kaggle-dr/blob/master/network_specs.json#L282) looked like:

| 9 Weight Layers         | Output Shape (batch size 128) |
|-------------------------|-------------------------------|
| input 256x256x3         | (3, 256, 256, 128)            |
| conv3-32                | (32, 254, 254, 128)           |
| maxpool size 3 stride 2 | (32, 127, 127, 128)           |
| conv3-64                | (64, 125, 125, 128)           |
| maxpool size 3 stride 2 | (64, 62, 62, 128)             |
| conv3-128               | (128, 60, 60, 128)            |
| conv3-128               | (128, 58, 58, 128)            |
| maxpool size 3 stride 2 | (128, 29, 29, 128)            |
| conv3-128               | (128, 27, 27, 128)            |
| maxpool size 3 stride 2 | (128, 13, 13, 128)            |
| conv3-128               | (256, 11, 11, 128)            |
| maxpool size 3 stride 2 | (256, 6, 6, 128)              |
| FC-2048                 | (128, 2048)                   |
| maxpool size 2          | (128, 1024)                   |
| FC-2048                 | (128, 2048)                   |
| maxpool size 2          | (128, 1024)                   |
| FC-4                    | (128, 4)                      |
| sigmoid                 |                               |

This network had [21.66 million parameters](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L101) and used up almost all of my availble 12Gb of GPU RAM. This same network with an input image of size 192x192 would fit in 6Gb of GPU RAM. Each convolutional layer (except the first) has dropout with *p=0.1*, and every convolutional layer has an LReLu non-linearity, and each FC layer has dropout with *p=0.5*.

The other successful network in [`network_specs.json`](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L387) is similar, but would take smaller input images, and has 1 fewer convolution layer.

The initial network decisions were made with the help of [Stanford tutorial][karpathy].

## Parameter Sharing Attempts

I tried the same solution Sander Dieleman did in his [plankton][plankton] challenge: splitting the image into quarters, running each quarter through the convolutional layers independently, and then connecting the features from the four quarters to the same series of 3 FC layers. I called this [`Fold4xBatchesLayer`](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/layers.py#L23) in the code because the pixels are being folded across the batch dimension in the 4D input data tensor. This marginally worsened results. Given more time, I would have combined this folding pixels across batches with the prealignment strategy.

I also tried a similar strategy but with [folding pixels across channels](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/layers.py#L8), which led to a considerable runtime speedup but 10% worse performance.

I would have also liked to try sharing parameters between pairs of images (left right). One problem is that not all image pairs share the same diagnosis.

## [Error Function](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L150) and Number of Output Nodes

I started by using the same error function I did for MNIST: [categorical cross entropy](http://lasagne.readthedocs.org/en/latest/modules/objectives.html?highlight=entropy#lasagne.objectives.categorical_crossentropy). This has the downside of not encoding any information that the classes are ordinal (4>3>2>1>0 in the severity of DR) and not differentiating between errors of different magnitudes (unlike the metric that the competition is judged on).

For this reason I followed the advice on the [Kaggle forums](https://www.kaggle.com/c/diabetic-retinopathy-detection/forums/t/13115/paper-on-using-ann-for-ordinal-problems) and used an nn-rank target matrix with relative entropy as described in [this paper](https://web.missouri.edu/~zwyw6/files/rank.pdf). This resulted in the [following target matrix](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L171) for a four node ouput network:

```
[
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
]
```

instead of the standard ["one-hot" target matrix](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L162) (which requires an additional output node):

```
[
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
]
```

I also tried an nn-rank target matrix with a 5 node output, but did not get superiour results.

## Variants

To emphasize gross errors, I [modified the target matrix](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L205) to be more punitive on errors of larger magnitude:

```
[
    [ 0,    0,    0,    0  ],
    [ 1,    0,    0,    0  ],
    [ 1.3,  0.6,  0,    0  ],
    [ 1.5,  1,    0.5,  0  ],
    [ 1.6,  1.2,  0.8,  0.4]
]
```

When I tried matricies that were [more steep](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L214) (concentrated most of the weight on more discrepant label-prediction pairs), I noticed that the network made fewer underestimates, and more overestimates. I saw the opposite effect as I tried [more flat matricies](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L196) (somewhere between the last one shown and the binary nn-rank target matrix). The best performing target matrix I tried is shown here and comes from the code [here](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L205).

If I had more time, my next step would have been to experiment directly with exagerating the underestimate penalty and under-emphasizing the overestimate penalty to exert more control over the false-negative <-> false-positive tradeoff.

# Training/Optimization

## Batch Selection

Each minibatch had the [same proportions of labels](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/block_designer.py#L126) as the entire training set. This led to a 2 to 3% performance improvement, and reduced the noise in the training error.

## Validation

[I laid away 15%](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/data_stream.py#L146) of the training set for validation, and held this set constant for my last 100 runs. Luckily, my validation set score was always ±0.5% of my Kaggle Public leaderboard submissions.

## SGD

## [Init](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L262)

Initialization proved important. [Normal](http://lasagne.readthedocs.org/en/latest/modules/init.html?highlight=normal#lasagne.init.Normal) initialization prevented training altogether, and [GlorotUniform](http://lasagne.readthedocs.org/en/latest/modules/init.html?highlight=normal#lasagne.init.GlorotUniform) worked best.

## [Learning Rate Decay](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L376)

Decaying the learning rate after the validation error stalled for 3 consecutive epochs led to minor improvements of 1 to 2%.

## Regularization

Very few experiments were run with [L1](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L173) and [L2](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L177) regularization. L1 always did considerable damage, and L2 didn't seem to help. With more time, I would have explore this.

## [Ensemble](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/avg_raw_outputs.py#L18)

I got about 1 to 2% improvements when I averaged the raw outputs of several networks (last 4 nodes after the nonlinearity for each example) and then made my prediction.

# Misc

## List of improvements (Left to Right on Summary graph)

*Numbers correspond to the experiment number in the first summary graph that each cumulative change corresponds to.*

13. vgg_mini7
14. GlorotUniform Init
16. All Conv dropout
18. Both FC pooling
19. 1 more FC dropout
21. Overlap Pooling
40. LReLu
55. [controlled batch distributions](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/block_designer.py#L8)
80. [nnrank-re](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L171)
83. color
91. 4 outputs
93. [random flips](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/data_stream.py#L15)
120. ±20 [color cast](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/data_stream.py#L69)
122. ±30 color cast
136. [kappa weighted error func](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/VGGNet.py#L205)
149. 152px
151. 192px + extra Pool
152. 192px + extra ConvPool
162. 256px

## [Dealing with Class Imbalance](https://github.com/ilyakava/kaggle-dr/blob/master/my_code/sampler.py#L26)

Since this is a real world medical dataset, it can be expected that pathological cases are greatly outnumbered by healthy ones. In this case, the class proportions are: `[0.73, 0.07, 0.15, 0.02, 0.02]` for classes `[0,1,2,3,4]`. I tried changing the class proportions in each minibatch to be more uniform. As I leveled out the populations, performance worsened as a result of the majority of the 0 labels being classified as pathological. This underperformance on class 0 was likely a result of undersampling that class, while overfitting from oversampling the rarer pathological cases set in.

I also tried training the network to discriminate the hardest classes first (only training on 0,1 examples, then only training on 0,1,2 examples, and so on), but this led to no performance increase either.

## Batch Size

I didn't experiment much with the batch size (stuck with 128 throughout), but would have liked to effectively increase it over the duration of the experiment (by accumulating gradients across minibatches before updating) to have less noise in the gradient steps.

## [Thinner Conv Layers](https://github.com/ilyakava/kaggle-dr/blob/master/network_specs.json#L250)

Thinning out the CONV layers led to 2 to 3% worse performance, but could reduce memory consumption and runtime by 1.5-2x. Compensating with additional input image resolution would have been nice.

# Conclusion

I had a lot of fun, and am thankful to Kaggle and the Sponsors for making such an exciting and challenging dataset available. Next steps for me will be experimenting with even larger images (I [had trouble](https://groups.google.com/d/msg/pylearn-users/vjtyydH8T8w/96Hgeee61gcJ) above 256px with my current setup), writing my own Cuda code for fun and potential performance gains, and maybe experimenting with other frameworks.

[GH]: https://github.com/ilyakava/kaggle-dr
[galaxy]: http://benanne.github.io/2014/04/05/galaxy-zoo.html
[plankton]: http://benanne.github.io/2015/03/17/plankton.html
[KL]: https://www.kaggle.com/c/diabetic-retinopathy-detection/leaderboard
[NIH]: https://nei.nih.gov/health/diabetic/retinopathy
[karpathy]: http://cs231n.github.io