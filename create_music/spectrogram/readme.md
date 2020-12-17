# Music Creation Neural Network

Builds a network using an input of switches.

Based on the approach taken making garfield comic strips be [CodeParade][1]
Source code: https://github.com/HackerPoet/Avant-Garfield

Samples taken from https://freemusicarchive.org/genre/Lo-fi

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


# Conclusion

For this first pass a spectrogram is used with a window length of 2048 noting the limitation of this approach and should the requirement to take the model to production a smaller window will be selected. 

# References

[1]: https://www.youtube.com/watch?v=wXWKWyALxYM
[2]: https://medium.com/better-programming/how-to-build-a-deep-audio-de-noiser-using-tensorflow-2-0-79c1c1aea299
[mel_scale_paper]: https://archive.is/20130414065947/http://asadl.org/jasa/resource/1/jasman/v8/i3/p185_s1
[tds_mel_spectrogram]: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
[mel_scale]: https://en.wikipedia.org/wiki/Mel_scale
[gla]: https://paperswithcode.com/method/griffin-lim-algorithm

[spectrogram]: https://upload.wikimedia.org/wikipedia/commons/2/29/Spectrogram_of_violin.png