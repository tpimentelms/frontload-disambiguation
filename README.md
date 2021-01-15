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

## Get Data

#### CELEX

CELEX data can be obtained at https://catalog.ldc.upenn.edu/LDC96L14/.
You can process it with the command:
```bash
$ make LANGUAGE=<language> DATASET=celex
```
Languages: eng, deu, nld.

#### NorthEuraLex

NorthEuraLex data already comes with this repo. To preprocess it, run:
```bash
$ make LANGUAGE=<language> DATASET=northeuralex
```
with any language in NorthEuraLex, e.g. `por`.

#### Wikipedia

To get the wikipedia tokenized data use the code in the [Wikipedia Tokenizer repository](https://github.com/tpimentelms/wiki-tokenizer).



## Train and evaluate the models

You can train your models using random search with the command
```bash
$ make LANGUAGE=<language> DATASET=<dataset>
```
There are three datasets available in this repository: celex; northeuralex; and wikipedia.
To get the wikipedia tokenized data use the code in the [Wikipedia Tokenizer repository](https://github.com/tpimentelms/wiki-tokenizer).


To train the model in all languages from one of the datasets, run
```bash
$ python src/h02_learn/train_all.py --dataset <dataset> --data-path data/<dataset>/
```

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
