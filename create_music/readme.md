# Music Creation Neural Network

Builds a network using an input of switches.

Based on the approach taken making garfield comic strips be [CodeParade][1]
Source code: https://github.com/HackerPoet/Avant-Garfield

Samples taken from https://freemusicarchive.org/genre/Lo-fi

# Training On Linear Final Output

The target variable of the network is the input sound files from librosa.
This model provides some hints at working and was accurately returning the original clip when feeding in a single variable but showed little sign of returning anything but noise when feeding multiple non-zero input.
Additionally, the model was too large to quickly hand around and deploying in the future would be complicated so another approach was taken before finalising.

# Training On Spectrogram

Target is spectrogram of the input sound file which is transformed back into normal sound files at the end.
This approach was taken since we can compress the information into a smaller dataset to keep the size of the final model smaller.
This approach was inspired by the article by [Daitan][2] where they built a network for removing noise from a sound file.


# References

[1]: https://www.youtube.com/watch?v=wXWKWyALxYM
[2]: https://medium.com/better-programming/how-to-build-a-deep-audio-de-noiser-using-tensorflow-2-0-79c1c1aea299


