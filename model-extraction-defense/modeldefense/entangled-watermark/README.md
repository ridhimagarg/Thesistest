# Entangled Watermarks as a Defense against Model Extraction

This repository is an implementation of the paper [Entangled Watermarks as a Defense against Model Extraction](https://arxiv.org/abs/2002.12200),	published in 30th USENIX Security Symposium. In this repository, we show how to train a watermarked DNN model that is robust against model extraction. The high-level idea is that a special watermark is designed such that it could be used to verify the owenrship of the model if it is stolen by model extraction. For more details, please read the paper.

* [TensorflowV2](./Tensorflowv2/)

    * [configs/](./Tensorflowv2/configs) - this contains all the configuration files

    These config files contains the by default parameters to be called in the each of python files. You can edit the parameters in these files only to be directly called in main python files.
    
    #### Below are all the primary code files (with all the descriptions already provided in each and every file, even the order of the execution is also provided with the filename.)

      Each of these files can be run as follows (for eg:) -:

        ```
        python ewe_train_retraining.py
        ```

    * [ewe_extraction.py](./Tensorflowv2/ewe_extraction.py) - 4th, model stealing attack with the perfect attacker setting (full data access and the exact model architecrure)
    * [ewe_train_finetuning.py](./Tensorflowv2/ewe_train_finetuning.py) - 3rd, embedding watermark with finetuning approach.
    * [ewe_train_retraining.py](./Tensorflowv2/ewe_train_retraining.py) - 3rd, embedding watermark with retraining approach.
    * [models_new.py](./Tensorflowv2/models_new.py) - contains all models.
    * [models_training_new.py](./Tensorflowv2/models_training_new.py) - (called by ewe_train_retraining.py or ewe_train_finetuning.py) logic of how to embed the watermark using entangle watermarking idea with SNNL
    * [real_model_stealing.py](./Tensorflowv2/real_model_stealing.py) - 4th performing KnockoffsNet
    * [setup.py](./Tensorflowv2/setup.py) - setup to run the folder as package.
    * [train_original.py](./Tensorflowv2/train_original.py) - 1st, training the original model to watermark it.
    * [trigger_new.py](./Tensorflowv2/trigger_new.py) - (called by ewe_train_retraining.py or ewe_train_finetuning.py) generating the watermark set to embed the watermark on the model.
    * [utils_new.py](./Tensorflowv2/utils_new.py) - (called internally) contains utility functionality.
    * [visualization.py](./Tensorflowv2/visualization.py)
    * [visualization_2.py](./Tensorflowv2/visualization_2.py)
    * [watermark_dataset.py](./Tensorflowv2/watermark_dataset.py) - (called by watermark_generate.py or watermark_generate_multiple.py)selecting the dataset to create watermark from.
    * [watermark_generate.py](./Tensorflowv2/watermark_generate.py) - 2nd, generating the watermark set with one source and target class.
    * [watermark_generate_multiple.py](./Tensorflowv2/watermark_generate_multiple.py) - 2nd, generating the watermark set with combination of source and target class.



