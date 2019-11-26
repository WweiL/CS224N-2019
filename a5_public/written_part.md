# Part 1
## (a)
Because there is much more word than characters in English. As a result, a character's neighbor is more fixed compared to words.

## (b)
 - Figure 2
  - V_{char} * e_char
  - Weight Matrix in the convolutional layer: f * e_{char} * k
  - Highway layer: 2 * (e_{word} ^ 2 + e_{word})
Total: 137384 (k = 4)
 - Figure 1
  - V_{word} * e_word
Total: 1280000

The word vector model has more parameters and is about 100 times more that what in the char vector model.

## (c)
CNN enables people to use different channel to extract different information. While in RNN there is no such a way. This makes sense in that in this char model, word is broke into characters that contains less information, more similar to pixels in image. Therefore, CNN is also a valid way of doing this job and it maybe more suitable.

## (d)
Max pooling focus more on the extreme values while average pooling focus on the whole context.
Therefore max pooling has an advantage of extracting most meaningful part which is something impossible for average pooling.
But average pooling can take care of the whole content, while max pooling can only focus on the peak values.
