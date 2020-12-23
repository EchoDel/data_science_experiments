# Music Creation Neural Network

Builds a network using an input of switches.

Based on the approach taken making garfield comic strips be [CodeParade][1]
Source code: https://github.com/HackerPoet/Avant-Garfield

Samples taken from https://freemusicarchive.org/genre/Lo-fi and https://github.com/mdeff/fma

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

## Research on the modelling problem

Pytorch seems to provide 2 layers which can be used as a solution to this the [Pixel Shuffle][pixel_shuffle] and [upsample][upsample_layer]

Research around this problem appears to be focused on upsampling images to higher resolutions while maintaining the visual fidelity.
Research by [Bee Lim et Al.][image_upsampling] uses a single upsampling layer for increasing image size by two to three times, and an additional layer when upscaling by four time.
Unfortunately for this application we are upsampling by 250 times with the current linear first layer, although the same level of detail is not needed.
We could introduce additional linear layers to reduce the upsampling ratio is lower but this will increase the size of the model drastically for minimal benefit compared to careful crafted convolution layers.

Since this relies on a late upsampling layer, which would need to be very large, this would put a large focus on a few inputs focused in the local area which would limit the accuracy of this approach due to the nature of the spectrogram.
Additionally, the detail of the input tensor is not required for the final output so there is limited benefits for this approach although having the passthrough and feature creation before the upsampling layer could still be used.

[Wenzhe Shi et al][pixel_shuffle_paper] introduced a new layer called the pixel shuffle to provide a quick layer for transposing between a sparse layer and the final output.
This layer rearranges elements in a tensor of shape (∗,C×r2,H,W) to a tensor of shape (∗,C,H×r,W×r) per the image below.

![Pixel Shuffle](https://github.com/redparry/data_science_experiments/blob/master/create_music/spectrogram/contents/model_1_architecture.png "Pixel Shuffle")


The main advantage of this method as stated in the paper is not requiring any layers to be in the high resolution space instead relying on smaller convolution layers.
Based on the conclusions from this paper this layer should be able to provide the required outputs and if the value of r is carefully crafted based on the input spectrogram then we could see very promising results.

## Experimenting with the r value

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

## Reviewing an optimal r value






# References

[1]: https://www.youtube.com/watch?v=wXWKWyALxYM
[2]: https://medium.com/better-programming/how-to-build-a-deep-audio-de-noiser-using-tensorflow-2-0-79c1c1aea299
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