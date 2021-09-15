# alignment-scripts
Scripts to preprocess training and test data for alignment experiments and to run and evaluate FastAlign and Mgiza.


## Dependencies
* Python3
* [MosesDecoder](https://github.com/moses-smt/mosesdecoder): Used during preprocessing
* [Sentencepiece](https://github.com/google/sentencepiece): Optional, used for subword splitting at the end of preprocessing
* [FastAlign](https://github.com/clab/fast_align): Only used for FastAlign
* [Mgiza](https://github.com/moses-smt/mgiza/): Only used for Mgiza


## Usage Instructions
* Install all necessary dependencies
* Export install locations for dependencies: `export {MOSES_DIR,FASTALIGN_DIR,MGIZA_DIR}=/foo/bar`
* Make sure you set a reasonable default locale, e.g.: `export LC_ALL=en_US.UTF-8`
* Create folder for your test data: `mkdir -p test`
* Download [Test Data for German-English](https://www-i6.informatik.rwth-aachen.de/goldAlignment/) and move it into the folder `test`
* Run preprocessing: `./preprocess/run.sh`
* Run Fastalign: `./scripts/run_fast_align.sh`
* Run Giza: `./scripts/run_giza.sh` (This might take multiple days)


## Results
All results are in percent in the format: AlignmentErrorRate (Precision/Recall)

### German to English ###
| Method | DeEn | EnDe | Grow-Diag | Grow-Diag-Final |
| --- | ---- | --- | ---- | --------- |
| FastAlign | 28.4% (71.3%/71.8%) | 32.0% (69.7%/66.4%) | 27.0% (84.6%/64.1%) | 27.7% (80.7%/65.5%) |
| Mgiza | 21.0% (86.2%/72.8%) | 23.1% (86.6%/69.0%) | 21.4% (94.3%/67.2%) | 20.6% (91.3%/70.2%) |

### Romanian to English ###
| Method | RoEn | EnRo | Grow-Diag | Grow-Diag-Final |
| --- | ---- | --- | ---- | --------- |
| FastAlign | 33.8% (71.8%/61.3%) | 35.5% (70.6%/59.4%) | 32.1% (85.1%/56.5%) | 32.2% (81.4%/58.1%) |
| Mgiza | 28.7% (82.7%/62.6%) | 32.2% (79.5%/59.1%) | 27.9% (94.0%/58.5%) | 26.4% (90.9%/61.8%) |

### English to French ###
| Method | EnFr | FrEn | Grow-Diag | Grow-Diag-Final |
| --- | ---- | --- | ---- | --------- |
| FastAlign | 16.4% (80.0%/90.1%) | 15.9% (81.3%/88.7%) | 10.5% (90.8%/87.8%) | 12.1% (87.7%/88.3%) |
| Mgiza | 8.0% (91.4%/92.9%) | 9.8% (91.6%/88.3%) | 5.9% (97.5%/89.7%) | 6.2% (95.5%/91.6%) |


## Known Issues
* Does not work on MacOs
* Tokenization of the Canadian Hansards seems to be off when accents are present in the English text: `Ms. H é l è ne Alarie`, `Mr. Andr é Harvey :`, `Mr. R é al M é nard`
