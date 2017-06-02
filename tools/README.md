# Tools

This directory contains additional tools.

## Generate Vocabulary

To generate the vocabulary:

```
python tools/generate_vocab.py -data_path data/train.txt -label_path data/labels.txt -vocab_file data/vocab.txt
```

where the options are:

* `-data_path`: Input file containing <image_path> <label_index> per line. This should be the file used for training.
    
* `-label_path`: Input file containing a tokenized formula per line.

* `-vocab_file`: Output file for the generated vocabulary. One token per line.
    
* `-unk_threshold`: If a token appears less than (including) the threshold, then it will be excluded from the vocabulary.
