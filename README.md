# Music Voice Separation

## Description

- Python package to generate vocal and instrumental tracks from original mix

## Data Structure for training/testing

All data must be in data directory

MusDB data shall be in musdb directory, ccMixter - in ccmixter_corpus

Construction of musdb and ccmixter_corpus must be as in original dataset

## Dependencies

Use python 3.12

To install dependencies use:
```shell
pip install -r requirements.txt
```


## TODO

- library interface (DONE)
- mocked functions (DONE)
- data preprocessing and loading using tf.data.Dataset (DONE)
- model training
- model output further processing (concatenating etc.)
- output evaluation (including Spleeter comparison)

mb add more points or split smthng
