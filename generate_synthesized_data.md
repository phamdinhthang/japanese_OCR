## Generate of synthesized data and data augmentation
An engine to generate synthesized data is provided to enhanced the performance of the system by adding more training sample to the training dataset. Synthesized samples were created using image rendering from Unicode character font. Moreover, data augmentation is also performed to increase the number of traning sample.

Script to generate synthesized data

```
cd /synthesized_training_data_and_augmentation/
python generate_image_data.py char_set mode image_size augmentation font_path
```

There are 5 arguments for the script:
* char_set: to simplify the selection of japanese character, 5 character set were provided: `hiragana:('3040','309f')` - the hiragana table, `katakana:('30a0','30ff')` - the katakana table, `kanji:('4e00','9faf')` - the full kanji table (note: over 20 000 characters), `kanji_1000:('4e00','5300')` - the first 1000 character of kanji table, `kanji_5000:('4e00','61ff')` - the first 5000 character of kanji table
* mode: color mode of the image: `rgb` or `grayscale`
* image_size: size of the image (square image, so width=height)
* augmentation: number of augmentation per character image per font (augmentation methods: random rotate, skew, distort, scale)
* font_path: path to the folder contains japanese unicode font