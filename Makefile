LANGUAGE := af
DATASET := wiki
MODEL := cloze
KEEP_EOS := False
N_PERMUT := 100000

DATA_DIR_BASE := ./data
DATA_DIR := $(DATA_DIR_BASE)/$(DATASET)
DATA_DIR_LANG := $(DATA_DIR)/$(LANGUAGE)
# WIKI_DIR := $(DATA_DIR_LANG)
CHECKPOINT_DIR_BASE := ./checkpoint
CHECKPOINT_DIR := $(CHECKPOINT_DIR_BASE)/$(DATASET)
CHECKPOINT_DIR_LANG := $(CHECKPOINT_DIR)/$(LANGUAGE)
RESULTS_DIR_BASE := ./results
RESULTS_DIR := $(RESULTS_DIR_BASE)/$(DATASET)
RESULTS_DIR_LANG := $(RESULTS_DIR)/$(LANGUAGE)

NORTHEURALEX_URL := http://www.sfs.uni-tuebingen.de/~jdellert/northeuralex/0.9/northeuralex-0.9-forms.tsv
NORTHEURALEX_RAW_FILE := $(DATA_DIR_BASE)/northeuralex/northeuralex-0.9-forms.tsv

CELEX_RAW_DIR := $(DATA_DIR_BASE)/celex/raw/
CELEX_EXTRACTED_DIR := $(CELEX_RAW_DIR)/extracted/
CELEX_EXTRACTED_FILE := $(CELEX_EXTRACTED_DIR)/$(LANGUAGE)_lemma_True_False_0_inf.tsv
CELEX_RAW_DIR_UNCOMPRESSED := $(CELEX_RAW_DIR)/LDC96L14/
CELEX_RAW_FILE := $(CELEX_RAW_DIR_UNCOMPRESSED)/extracted.txt
CELEX_RAW_FILE_COMPRESSED := $(CELEX_RAW_DIR)/LDC96L14.tar.gz

WIKI_TOKENIZED_FILE := $(DATA_DIR_BASE)/wiki/$(LANGUAGE)/parsed.txt

PROCESSED_NORTHEURAGRAPH_FILE := $(DATA_DIR_BASE)/northeuragraph/$(LANGUAGE)/processed.pckl
PROCESSED_NORTHEURALEX_FILE := $(DATA_DIR_BASE)/northeuralex/$(LANGUAGE)/processed.pckl
PROCESSED_WIKI_FILE := $(DATA_DIR_BASE)/wiki/$(LANGUAGE)/processed.pckl
PROCESSED_CELEX_FILE := $(DATA_DIR_BASE)/celex/$(LANGUAGE)/processed.pckl
PROCESSED_SYLLEX_FILE := $(DATA_DIR_BASE)/syllex/$(LANGUAGE)/processed.pckl
PROCESSED_DATA_FILE := $(DATA_DIR_LANG)/processed.pckl

CHECKPOINT_FILE := $(CHECKPOINT_DIR_LANG)/norm/model.tch
CHECKPOINT_FILE_REVERSED := $(CHECKPOINT_DIR_LANG)/rev/model.tch
CHECKPOINT_FILE_UNIGRAM := $(CHECKPOINT_DIR_LANG)/unigram/model.tch
CHECKPOINT_FILE_CLOZE := $(CHECKPOINT_DIR_LANG)/cloze/model.tch
CHECKPOINT_FILE_POSITION_NN := $(CHECKPOINT_DIR_LANG)/position-nn/model.tch

LOSSES_FILE := $(CHECKPOINT_DIR_LANG)/norm/losses.pckl
LOSSES_FILE_REVERSED := $(CHECKPOINT_DIR_LANG)/rev/losses.pckl
LOSSES_FILE_UNIGRAM := $(CHECKPOINT_DIR_LANG)/unigram/losses.pckl
LOSSES_FILE_CLOZE := $(CHECKPOINT_DIR_LANG)/cloze/losses.pckl
LOSSES_FILE_POSITION_NN := $(CHECKPOINT_DIR_LANG)/position-nn/losses.pckl

P_VALUES_DIR := $(RESULTS_DIR_BASE)/p_values
P_VALUES_FILE_BIN := $(P_VALUES_DIR)/bin--$(DATASET)_$(MODEL)__$(KEEP_EOS)--$(N_PERMUT).tsv

all: get_data train eval

p_values: $(P_VALUES_FILE_BIN)
	echo $(P_VALUES_FILE_BIN)
	python src/h04_analysis/print_significant_diffs.py --dataset $(DATASET) --model-type $(MODEL) --n-permutations $(N_PERMUT)

print_diffs:
	python src/h04_analysis/print_diff_table.py --n-permutations $(N_PERMUT)

print_eow:
	python src/h04_analysis/print_eow.py --data-path $(DATA_DIR_BASE) --checkpoints-path $(CHECKPOINT_DIR_BASE)

plot_first_page:
	python src/h04_analysis/plot_forward_backward.py --data-path $(DATA_DIR_BASE) --checkpoints-path $(CHECKPOINT_DIR_BASE) --results-path $(RESULTS_DIR_BASE)

plot_bin:
	python src/h04_analysis/plot_bin.py --dataset $(DATASET) --n-permutations $(N_PERMUT) --results-path $(RESULTS_DIR_BASE)


eval: $(LOSSES_FILE) $(LOSSES_FILE_REVERSED) $(LOSSES_FILE_UNIGRAM) $(LOSSES_FILE_POSITION_NN) $(LOSSES_FILE_CLOZE)
	echo "Finished evaluating model" $(LANGUAGE)

train: $(CHECKPOINT_FILE) $(CHECKPOINT_FILE_REVERSED) $(CHECKPOINT_FILE_UNIGRAM) $(CHECKPOINT_FILE_POSITION_NN) $(CHECKPOINT_FILE_CLOZE)
	echo "Finished training model" $(LANGUAGE)

train_cloze: $(CHECKPOINT_FILE_CLOZE) $(CHECKPOINT_FILE_POSITION_NN)
	echo "Finished training model" $(LANGUAGE)

train_lstm: $(CHECKPOINT_FILE) $(CHECKPOINT_FILE_REVERSED)
	echo "Finished training model" $(LANGUAGE)

get_data: $(PROCESSED_DATA_FILE)
	echo "Finished getting data" $(LANGUAGE)

get_northeuralex: $(NORTHEURALEX_RAW_FILE)
	echo "Getting northeuralex data"

get_celex: $(PROCESSED_CELEX_FILE)

clean:
	rm $(PROCESSED_DATA_FILE)


$(P_VALUES_FILE_BIN):
	echo $(P_VALUES_FILE_BIN)
	mkdir -p $(P_VALUES_DIR)
	python src/h04_analysis/p_values_bin.py --data-path $(DATA_DIR) --dataset $(DATASET) --checkpoints-path $(CHECKPOINT_DIR) --n-permutations $(N_PERMUT)

# Eval language models
$(LOSSES_FILE_POSITION_NN): | $(CHECKPOINT_FILE)
	echo "Eval models" $(LOSSES_FILE_POSITION_NN)
	python src/h03_eval/eval.py --dataset $(DATASET) --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type position-nn

# Eval language models
$(LOSSES_FILE_CLOZE): | $(CHECKPOINT_FILE)
	echo "Eval models" $(LOSSES_FILE_CLOZE)
	python src/h03_eval/eval.py --dataset $(DATASET) --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type cloze

# Eval language models
$(LOSSES_FILE_UNIGRAM): | $(CHECKPOINT_FILE)
	echo "Eval models" $(LOSSES_FILE_UNIGRAM)
	python src/h03_eval/eval.py --dataset $(DATASET) --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type unigram

# Eval language models
$(LOSSES_FILE_REVERSED): | $(CHECKPOINT_FILE)
	echo "Eval models" $(LOSSES_FILE_REVERSED)
	python src/h03_eval/eval.py --dataset $(DATASET) --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type rev

# Eval language models
$(LOSSES_FILE): | $(CHECKPOINT_FILE)
	echo "Eval models" $(LOSSES_FILE)
	python src/h03_eval/eval.py --dataset $(DATASET) --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type norm

# Train types Model
$(CHECKPOINT_FILE_POSITION_NN): | $(PROCESSED_DATA_FILE)
	echo "Train types model" $(CHECKPOINT_FILE_POSITION_NN)
	mkdir -p $(CHECKPOINT_DIR_LANG)/position-nn/
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type position-nn

# Train types Model
$(CHECKPOINT_FILE_CLOZE): | $(PROCESSED_DATA_FILE)
	echo "Train types model" $(CHECKPOINT_FILE_CLOZE)
	mkdir -p $(CHECKPOINT_DIR_LANG)/cloze/
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type cloze

# Fit unigram Model
$(CHECKPOINT_FILE_UNIGRAM): | $(PROCESSED_DATA_FILE)
	echo "Fit unigram model" $(CHECKPOINT_FILE_UNIGRAM)
	mkdir -p $(CHECKPOINT_DIR_LANG)/unigram/
	python src/h02_learn/fit.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type unigram

# Train types Model
$(CHECKPOINT_FILE_REVERSED): | $(PROCESSED_DATA_FILE)
	echo "Train types model reversed " $(CHECKPOINT_FILE_REVERSED)
	mkdir -p $(CHECKPOINT_DIR_LANG)/rev/
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type rev

# Train types Model
$(CHECKPOINT_FILE): | $(PROCESSED_DATA_FILE)
	echo "Train types model" $(CHECKPOINT_FILE)
	mkdir -p $(CHECKPOINT_DIR_LANG)/norm/
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type norm

#### Preprocess Data ####

$(PROCESSED_WIKI_FILE): | $(WIKI_TOKENIZED_FILE)
	echo "Process wiki data"
	python src/h01_data/process_data.py --dataset $(DATASET) --src-file $(WIKI_TOKENIZED_FILE) --data-path $(DATA_DIR_LANG)

$(PROCESSED_NORTHEURALEX_FILE): | $(NORTHEURALEX_RAW_FILE)
	echo "Process northeuralex data"
	python src/h01_data/process_data.py --dataset $(DATASET) --src-file $(NORTHEURALEX_RAW_FILE) --data-path $(DATA_DIR)

$(PROCESSED_NORTHEURAGRAPH_FILE): | $(NORTHEURALEX_RAW_FILE)
	echo "Process northeuralex data"
	python src/h01_data/process_data.py --dataset $(DATASET) --src-file $(NORTHEURALEX_RAW_FILE) --data-path $(DATA_DIR)

$(NORTHEURALEX_RAW_FILE):
	echo "Get northeuralex data"
	mkdir -p $(DATA_DIR)
	wget -P $(DATA_DIR) $(NORTHEURALEX_URL)

$(PROCESSED_CELEX_FILE) $(PROCESSED_SYLLEX_FILE): | $(CELEX_EXTRACTED_FILE)
	mkdir -p $(DATA_DIR_LANG)
	python src/h01_data/process_data.py --dataset $(DATASET) --src-file $(CELEX_EXTRACTED_FILE) --data-path $(DATA_DIR_LANG)

$(CELEX_EXTRACTED_FILE): | $(CELEX_RAW_FILE)
	mkdir -p $(CELEX_EXTRACTED_DIR)
	python src/h01_data/extract_lex_celex.py --language $(LANGUAGE) --src-path $(CELEX_RAW_DIR_UNCOMPRESSED) --tgt-path $(CELEX_EXTRACTED_DIR)

$(CELEX_RAW_FILE): | $(CELEX_RAW_FILE_COMPRESSED)
	echo "Get celex data"
	tar -C $(CELEX_RAW_DIR) -zxvf $(CELEX_RAW_FILE_COMPRESSED)
	touch $(CELEX_RAW_FILE)
