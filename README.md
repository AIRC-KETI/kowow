

## Install parlai package

```bash
python setup.py develop
```

## Download Korean Wizard of Wikipedia (KoWoW)

[Download](https://drive.google.com/drive/folders/1L_6WLUYkUtwOEOVC9KI6w4-xkW9ZkUKp?usp=sharing)

```bash
tar xvzf wizard_of_wikipedia_ko.tar.gz
mv wizard_of_wikipedia_ko data/
```

# language combinations

| Name | Knowledge | Utterance |
| ------------- | ------------- | ------------- |
| ko  | Korean  | Korean  |
| ke  | Korean  | English  |
| ek  | English  | Korean  |
| en  | English  | English  |

# Test splits
| Name | topic |
| ------------- | ------------- |
| random_split  | Seen  |
| topic_split  | Unseen  |


## training on KoWoW (ek dataset)

```bash
CUDA_VIS_DEV=0
LANG_TYPE=ek

NUM_EPOCHS=10
MODEL_TYPE=T5EndToEndTwoAgent
MODEL_NAME=mymodel

CUDA_VISIBLE_DEVICES="${CUDA_VIS_DEV}" parlai train_model -t wizard_of_wikipedia_ko:generator:topic_split --ln ${LANG_TYPE} -m projects.wizard_of_wikipedia_ko.generator.t5:${MODEL_TYPE}  -mf model_data/model/${MODEL_NAME}  --t5-model-arch "KETI-AIR/ke-t5-base" --t5-encoder-model-arch "KETI-AIR/ke-t5-small" --log-every-n-secs 10 --validation-patience 12 --validation-metric ppl --validation-metric-mode min --validation-every-n-epochs 5 -bs 4 --max_knowledge 32 --num-epochs ${NUM_EPOCHS} --tensorboard-log true --tensorboard-logdir ./model_data/tf_logs/${MODEL_NAME}
```

## test topic split(Unseen) on KoWoW ek dataset

```bash
CUDA_VIS_DEV=0
LANG_TYPE=ek
MODEL_NAME=mymodel

CUDA_VISIBLE_DEVICES="${CUDA_VIS_DEV}" parlai eval_model -t wizard_of_wikipedia_ko:generator:topic_split -dt test --ln ${LANG_TYPE} -mf model_data/model/${MODEL_NAME} -bs 4 --inference beam --beam-size 4
```

## Display sample - topic split(Unseen)

You can display predicted knowledge using `enc_output` field.

```bash
CUDA_VIS_DEV=0
LANG_TYPE=ek
MODEL_NAME=mymodel
NUM_EXAMPLES=300

CUDA_VISIBLE_DEVICES="${CUDA_VIS_DEV}" parlai display_model -t wizard_of_wikipedia_ko:generator:topic_split -dt test --ln ${LANG_TYPE} -mf ./model_data/model/${MODEL_NAME} -bs 1 --inference beam --beam-size 4 --display-add-fields checked_sentence,enc_output -n NUM_EXAMPLES
```
