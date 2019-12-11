# Description

Many image processing applications make use of digitalized textual data. However, the presence of any type of noise can create difficulties in post-processing information, such as on OCR detection. To improve the information manipulation on such data, a previous image processing step is required.

In light of this idea, a set of text paragraphs containing plain English language were collected. Different font styles, size, and background noise level were arranged to simulate the most probable situations on real-case scenarios.

# Objective

The objective of this test is to evaluate the possible image processing methods that could fix the text samples. Note that the samples have a different type of background noise and present a set of text fonts. Therefore, the candidate should provide a flexible algorithm that can correctly detect what is text characters and background noise, offering a clean version of each text paragraph.

# Important details

- As a common pattern, the text must be on BLACK pixels and the background as WHITE pixels. Therefore, the output image MUST be in binary format (i.e. `0` pixel values for text and `255` pixel values for background)
- The candidate will receive only the noisy data due to the presence of cleaned data is rarely provided on real-case scenarios. Thus, be creative for the intention to use supervised learning algorithms.
- This test does not require a defined image processing algorithm to be used. The candidate is free to choose any kind of image processing pipeline to reach the best answer.
- The considered correct data is when: 1) white background, 2) black text characters, 3) text horizontal aligment, 4) centered in the image and 5) the text font is straight (not itallic formatting).
- Do not change the filename when applying your image processing methods. The filename is important for data comparison purposes.
- The output image file will be only accepted on the following formats: `.png`, `.tif`, `.jpg`
