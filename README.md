# frontload-disambiguation

This code accompanies the paper "Disambiguatory signals are stronger in word initial positions" published in EACL 2021.


## Install

To install dependencies run:
```bash
$ conda env create -f environment.yml
```

And then install the appropriate version of pytorch:
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
```

## Train and evaluate the models

You can train your models using random search with the command
```bash
$ make LANGUAGE=<language> DATASET=<dataset>
```
There are three datasets available in this repository: celex; northeuralex; and wikipedia.
To get the wikipedia tokenized data use the code in the [Wikipedia Tokenizer repository](https://github.com/tpimentelms/wiki-tokenizer).


## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:

```bash
@inproceedings{pimentel-etal-2021-disambiguatory,
    title = "Disambiguatory signals are stronger in word initial positions",
    author = "Pimentel, Tiago and
    Cotterell, Ryan and
    Roark, Brian",
    booktitle = "Proceedings of the 16th Conference of the {E}uropean Chapter of the Association for Computational Linguistics: Volume 1, Long Papers",
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/tpimentelms/frontload-disambiguation/issues).
