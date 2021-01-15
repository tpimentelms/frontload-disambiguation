# frontload-disambiguation

[![CircleCI](https://circleci.com/gh/tpimentelms/frontload-disambiguation.svg?style=svg&circle-token=0849cab470f63aacaf87c631c0190887f7645284)](https://circleci.com/gh/tpimentelms/frontload-disambiguation)


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
There are three datasets available in this repository: celex; northeuralex; and wiki.


To train the model in all languages from one of the datasets, run
```bash
$ python src/h02_learn/train_all.py --dataset <dataset> --data-path data/<dataset>/
```

The names of the models in this repository differ from the paper. They are: `norm` (Forward); `rev` (Backward); `cloze` (Cloze); `unigram` (Unigram); `position-nn` (Position-specific).


## Print and Plot Paper Results

To make the first page plot (forward and backward surprisal plots) use command:
```bash
$ make plot_first_page
```

To get and print the p-values used in the statistical significance tests
```bash
$ make p_value MODEL=<model> DATASET=<dataset>
```

The command to plot Figures 2 and 3 is:
```bash
$ make plot_bin
```
the plots will be created in folder `results/`. Finally, to print Tables 2 and 3 run:
```bash
$ make print_eow
$ make print_diffs
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
