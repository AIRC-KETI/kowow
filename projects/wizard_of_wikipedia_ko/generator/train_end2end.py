#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.scripts.train_model import setup_args, TrainLoop

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='wizard_of_wikipedia_ko:generator:train',
        model='projects.wizard_of_wikipedia_ko.generator.t5:T5EndToEndAgent',
        model_file='/tmp/end2end_generator/model',
        t5_model_arch='pretrained_model/t5.1.1.base.gin_ke.ke_v100_span_corruption_600K',
        text_truncate=256,
        ln='ko',
        log_every_n_secs=10,
        validation_patience=12,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_every_n_epochs=0.5,
        truncate=256,
        max_knowledge=32,
        knowledge_alpha=0.95,
        knowledge_truncate=64,
        learningrate=5e-4,
        warmup_updates=5000,
        clip=0.1,
        lr_scheduler='invsqrt',
        embedding_type='fasttext',
        beam_size=1,
        skip_generation=False,
        batchsize=64,
    )
    TrainLoop(parser.parse_args()).train()


# parlai train_model -m projects.wizard_of_wikipedia_ko.generator.t5:T5EndToEndAgent -mf model/ke-t5_test -t wizard_of_wikipedia_ko:generator:random_split --ln en -bs 4 -eps 1 -lr 1e-5 --num-epochs 1 --optimizer adam --t5-model-arch pretrained_model/t5.1.1.base.gin_ke.ke_v100_span_corruption_600K --text_truncate 512
# parlai train_model -t wizard_of_wikipedia_ko:generator:random_split --ln ke_mix -m projects.wizard_of_wikipedia_ko.generator.t5:T5EndToEndAgent  -mf model/ke-t5_test --t5-model-arch ../pretrained_model/t5.1.1.base.gin_ke.ke_v100_span_corruption_600K --log-every-n-secs 10 --validation-patience 12 --validation-metric ppl --validation-metric-mode min --validation-every-n-epochs 0.5 -bs 4 --max_knowledge 32 --num-epochs 1 