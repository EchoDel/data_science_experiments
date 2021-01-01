# Music Creation Neural Network

Builds a network using an input of switches.

Based on the approach taken making garfield comic strips be [CodeParade][1]
Source code: https://github.com/HackerPoet/Avant-Garfield

Samples taken from https://freemusicarchive.org/genre/Lo-fi and [https://github.com/mdeff/fma][fma]

# Training On Spectrogram

Instead of training on the linear raw sound file if instead we transform the sound into a spectrogram we can reduce the model complexity for the sound file length.

This approach was inspired by the article by [Daitan][2] where they built a network for removing noise from a sound file.

# Spectrograms

A spectrogram breaks down the sound input into bins of frequency intensity using a fourier transform to present the signal as a three-dimensional heat map.
By transposing the input file into this format we can reduce the data required to represent this signal and therefore make any models built from this simpler.
Unfortunately this is a lossy transformation, so the output won't sound perfect but will be good enough for this experiment.

Spectrogram of a violin [from wikipedia](https://en.wikipedia.org/wiki/File:Spectrogram_of_violin.png)
![Spectrogram of a violin][spectrogram]

## Mel spectrogram

As humans, we don't hear sounds in the linear bands which traditional spectrograms use, instead it follows a more logarithmic banding. Mel spectrograms address this by binning the frequencies into in line with this logarithmic scale based on the research by [Stevens, Volkmann, and Newman][mel_scale_paper]

For more information on Mel Spectrograms refer to this [medium article][tds_mel_spectrogram] or the wikipedia page on the [mel scale][mel_scale]

## Inverting the spectrogram

In order to create an audible output after the modelling we need to inverse the spectrogram creation. Librosa provides a function which implements the [Griffin-Lim algorithim][gla]. 
This is an iterative method where successive iterations minimize the error between the current output and the spectrogram.

As you increase the number of iterations the result sound file better reflects the spectrogram but it takes time to run.
This also needs to be balanced with the time period of each bin (n_fft) in the spectrogram to balance the complexity of the model with the time taken to output a sound file at the end.
Tests were performed balancing these two variable to evaluate the;
 * length of the spectrogram,   
 * mean squared error against the source sound file
 * time take to recreate the sound file from the spectrogram

### Processing Time

When the window length is 128 and under samples the processing time is under 30 seconds which could be considered a reasonable time to generate. As we go to 1024 and above the processing time is 600 seconds or 10 minutes and up which would appear unreasonably slow.
Additionally, we can also see that the error increases as we increase the sample size, presumably since there is more information which can be used to recreate the signals.

![Image of the processing time against the window length](https://github.com/redparry/data_science_experiments/blob/master/create_music/spectrogram/contents/spectrogram_settings_time.png "Spectrogram Processing Time")

### Length of the Spectrogram

When comparing the length of the resulting spectrogram the amount of data needed to represent the signal decreases in a logarithmic fashion as you increase the windows length so picking a larger window will allow for a longer final sample with the same complexity of model.

![Image of the length of the sample sound file](https://github.com/redparry/data_science_experiments/blob/master/create_music/spectrogram/contents/spectrogram_settings_length.png "Spectrogram Output Length")


### Listening to the Quality

As the saying goes, the proof of the pudding is in the eating, so by listening to the outputs from this transformation we can draw more conclusions
Source:Urban Haze by Scott Holmes Music

#### Original
<audio controls>
  <source src="https://raw.githubusercontent.com/redparry/data_science_experiments/master/create_music/spectrogram/contents/gyNN33kV2jCi8mFtwMpHMEV9Hajbtc5XSrWxZzPg.mp3" type="audio/mp3">
Your browser does not support the audio element.
</audio> 

#### Spectrogram 64
<audio controls>
  <source src="https://raw.githubusercontent.com/redparry/data_science_experiments/master/create_music/spectrogram/contents/sample_audio_64.wav" type="audio/wav">
Your browser does not support the audio element.
</audio>

#### Spectrogram 2048
<audio controls>
  <source src="https://raw.githubusercontent.com/redparry/data_science_experiments/master/create_music/spectrogram/contents/sample_audio_2048.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> 

#### Mel Spectrogram 64
<audio controls>
  <source src="https://raw.githubusercontent.com/redparry/data_science_experiments/master/create_music/spectrogram/contents/sample_audio_2048.wav" type="audio/wav">
Your browser does not support the audio element.
</audio>

#### Mel Spectrogram 2048
<audio controls>
  <source src="https://raw.githubusercontent.com/redparry/data_science_experiments/master/create_music/spectrogram/contents/sample_audio_mel_2048.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> 

### Conclusion for spectrogram approach

For this first pass a spectrogram is used with a window length of 2048 noting the limitation of this approach and should the requirement to take the model to production a smaller window will be selected. 

# Model Development

## First Model
The first model was extended from the alexnet architecture with the simplistic approach of including a final layer which had a large padding to match the expected size.
This results in a nearly fully zero output which have no ability to be trained. 
As such this network will never be able to produce a suitable output, so a new architecture was created starting with the diagram and then transposed to code.

![Image of the length of the sample sound file](https://github.com/redparry/data_science_experiments/blob/master/create_music/spectrogram/contents/model_1_architecture.png "First Neural Network Diagram")

Diagram produced with [Alex le nails](http://alexlenail.me/NN-SVG/AlexNet.html) neural network tool.

## Second Model

With the failure of the first model research was conducted on what techniques others have used in literature to acheive similar results.

### Research on the modelling problem

Pytorch seems to provide 2 layers which can be used as a solution to this the [Pixel Shuffle][pixel_shuffle] and [upsample][upsample_layer]

Research around this problem appears to be focused on upsampling images to higher resolutions while maintaining the visual fidelity.
Research by [Bee Lim et Al.][image_upsampling] uses a single upsampling layer for increasing image size by two to three times, and an additional layer when upscaling by four time.
Unfortunately for this application we are upsampling by 250 times with the current linear first layer, although the same level of detail is not needed.
We could introduce additional linear layers to reduce the upsampling ratio is lower but this will increase the size of the model drastically for minimal benefit compared to careful crafted convolution layers.

Since this relies on a late upsampling layer, which would need to be very large, this would put a large focus on a few inputs focused in the local area which would limit the accuracy of this approach due to the nature of the spectrogram.
Additionally, the detail of the input tensor is not required for the final output so there is limited benefits for this approach although having the passthrough and feature creation before the upsampling layer could still be used.

[Wenzhe Shi et al][pixel_shuffle_paper] introduced a new layer called the pixel shuffle to provide a quick layer for transposing between a sparse layer and the final output.
This layer rearranges elements in a tensor of shape (∗,C×r2,H,W) to a tensor of shape (∗,C,H×r,W×r) per the image below.

![Pixel Shuffle](https://github.com/redparry/data_science_experiments/blob/master/create_music/spectrogram/contents/pixel_shuffle_image.png "Pixel Shuffle")


The main advantage of this method as stated in the paper is not requiring any layers to be in the high resolution space instead relying on smaller convolution layers.
Based on the conclusions from this paper this layer should be able to provide the required outputs and if the value of r is carefully crafted based on the input spectrogram then we could see very promising results.

### Experimenting with the r value

Expecting the final output to be 128 by 512, since the maths is both simpler now but also cleaner when transformed in to the network, and have a single channel we have a few options for r presented below.

| r  | w_in | h_in | c  |
|----|------|------|----|
| 2  | 256  | 64   | 2  |
| 4  | 128  | 32   | 4  |
| 8  | 64   | 16   | 8  |
| 16 | 32   | 8    | 16 |
| 32 | 16   | 4    | 32 |
| 64 | 8    | 2    | 64 |

None of these seem outlandish options but do encourage a non-square input from the linear layer. 

### Reviewing an optimal r value

By building a network where the value of r lines up with the features of the spectrogram we can reduce the complexity needed to represent the sound file.
Looking at a single slice in the y-axis of the spectrogram we can view how the frequencies behave, while looking at a slice in the x-axis we can view how a single frequency bin changes through time.

By applying a smoothing function we can isolate peaks in this frequency space and use this to find a value of R which assists in reducing complexity.

![R Value Selection Methodology](https://github.com/redparry/data_science_experiments/blob/master/create_music/spectrogram/contents/maximum_method.png "R Value Selection Methodology")

By applying this methodology to every song in the [FMA][fma] dataset we can review the spacing of the peak bins for each genre of music.

|                     | number of mel bins |           |           |
|---------------------|--------------------|-----------|-----------|
| Genre               | 128                | 256       | 376       |
| Blues               | 11.878658          | 17.203083 | 20.608692 |
| Classical           | 11.457404          | 16.681236 | 19.949109 |
| Country             | 11.440129          | 16.581485 | 19.871212 |
| Easy Listening      | 11.251914          | 17.077822 | 20.722105 |
| Electronic          | 11.749551          | 17.558706 | 20.939769 |
| Experimental        | 12.186012          | 17.854416 | 21.178206 |
| Folk                | 11.783293          | 16.667163 | 19.815720 |
| Hip-Hop             | 11.974557          | 17.551491 | 20.928376 |
| Instrumental        | 11.761550          | 17.211259 | 20.362515 |
| International       | 11.337527          | 16.966899 | 20.511056 |
| Jazz                | 11.673325          | 17.240927 | 20.722238 |
| Old-Time / Historic | 11.190302          | 16.637667 | 20.573574 |
| Pop                 | 11.639986          | 16.950981 | 20.220419 |
| Rock                | 11.865741          | 17.195298 | 20.414083 |
| Soul-RnB            | 11.976765          | 17.544161 | 20.891186 |
| Spoken              | 13.143180          | 18.741906 | 21.604005 |

There seems to be a small change between the number of bins selected, and the spacing of the maxima but this change seems to be mainly driven by the size of the window function.

Based on this the upscaling layer will be used to reshape a tensor of shape (N, 512, 17, 17) into a tensor of shape (N, 1, 512, 289) using a technique akin to pixel shuffle but rewritten to take the right inputs and outputs as required.
This will then have an additional convolution layer which reduces this into a tensor of shape (N, 1, 500, 256) output layer.

### Model Design

#### First layer
This layer takes the single bit inputs and returns a common shape of tensor which is reshaped into a rectangle to feed into the convolutional layers.
With 25000 songs in the sample, and a final output before the upsampling of 17 by 17, we require a reasonably large linear layer to capture the required information.
For this reason a starting point for this layer will be a 25000 inputs and 4096 output. This equates to a 32 by 32 once it has been reshaped.

For now this layer is transformed simply using the tensor.view function from pytorch but this may be more efficient if a different shaping layer is used.

#### Feature Creation Layers

This layer will need to transform the (N, 1, 32, 32) tensor into a (N, 512, 17, 17) tensor through a series of convolutions.
Since this is an akin to an image upsampling task where all the detail is required the layers from the [Bee Lim et Al.][image_upsampling] paper were taken as insperation, but the original input is not required so no addition step will be applied.

Since we are going from one channel to 512 channels 5 convolution layers were selected with dropouts after the first and third layer.

#### Upsampling

Based on the experiments in the script upsampling_layer_test.py the [Flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html) layer performs the correct reshape of the input tensor so has been selected for now.
This layer will unwrap the final two layers of the tensor into a single lien in the output tensor.

#### Output Layer

The layer will take the input tensor of shape of (1, 512, 289), and needs to return a tensor of shape (1, 512, 256). 
Using a 2d convolution layer we can calculate the possible options for the parameters of this layer using the equation below

![Pytorch convolution shape function](https://discuss.pytorch.org/uploads/default/optimized/2X/f/fab7651734d4000309a49736e88fc8e8b0bc2221_2_690x151.png "PyTorch convolution layer shape formulae")

Knowing that p will be zero since we require all the information present at the end, and s will be 1 since we don't want to reduce the size of the input tensor much we can reduce this formulae to;

![33=kd-d](https%3A%2F%2Flatex.codecogs.com%2Fsvg.latex%3F%5CLarge%26space%3B33%3Dkd-d "33=kd-d")

| Parameter | 1   | 2   | 3   | 4   |
|-----------|-----|-----|-----|-----|
| Hin       | 289 | 289 | 289 | 289 |
| Hout      | 256 | 256 | 256 | 256 |
| s         | 1   | 1   | 1   | 1   |
| p         | 0   | 0   | 0   | 0   |
| k         | 2   | 4   | 12  | 34  |
| d         | 33  | 11  | 3   | 1   |

Performing the same calculation on the width the formulae is reduced to;

![12=kd-d](https%3A%2F%2Flatex.codecogs.com%2Fsvg.latex%3F%5CLarge%26space%3B12%3Dkd-d "12=kd-d")

| Parameter | 1   | 2   | 3   | 4   | 5   | 6   |
|-----------|-----|-----|-----|-----|-----|-----|
| Win       | 512 | 512 | 512 | 512 | 512 | 512 |
| Wout      | 500 | 500 | 500 | 500 | 500 | 500 |
| s         | 1   | 1   | 1   | 1   | 1   | 1   |
| p         | 0   | 0   | 0   | 0   | 0   | 0   |
| k         | 13  | 7   | 5   | 4   | 3   | 2   |
| d         | 1   | 2   | 3   | 4   | 6   | 12  |

In the future this equation would be incorporated into the class so it and a pair would be selected based on experiments to increase how generic this model is.

For now the second lowest value of d was selected, and the corresponding k value. 

d = (3, 2)

k = (12, 7)

# References

[1]: https://www.youtube.com/watch?v=wXWKWyALxYM
[2]: https://medium.com/better-programming/how-to-build-a-deep-audio-de-noiser-using-tensorflow-2-0-79c1c1aea299
[fma]: https://github.com/mdeff/fma
[mel_scale_paper]: https://archive.is/20130414065947/http://asadl.org/jasa/resource/1/jasman/v8/i3/p185_s1
[tds_mel_spectrogram]: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
[mel_scale]: https://en.wikipedia.org/wiki/Mel_scale
[gla]: https://paperswithcode.com/method/griffin-lim-algorithm
[upsampling_article]: https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
[upsample_layer]: https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
[pixel_shuffle]: https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
[image_upsampling]: https://arxiv.org/abs/1707.02921
[pixel_shuffle_paper]: https://arxiv.org/pdf/1609.05158v2.pdf

[spectrogram]: https://upload.wikimedia.org/wikipedia/commons/2/29/Spectrogram_of_violin.png

Tables generated using [Tables Generator](https://www.tablesgenerator.com/markdown_tables)
