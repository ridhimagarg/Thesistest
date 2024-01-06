# Frontier-Stitching and Extended Frontier Stitching (Proposed Method)

## Directory Structure and Files Description
* [code/](./code/)
  * [configs/](./code/configs) -- this contains all the configuration files
    * [fgsm.yaml](./code/configs/fgsm.yaml)
    * [knockoffattack_finetuned.yaml](./code/configs/knockoffattack_finetuned.yaml)
    * [knockoffattack_original.yaml](./code/configs/knockoffattack_original.yaml)
    * [original.yaml](./code/configs/original.yaml)
    * [pruning.yaml](./code/configs/pruning.yaml)
    * [watermarking_finetuning.yaml](./code/configs/watermarking_finetuning.yaml)
    * [watermarking_finetuning_freezing.yaml](./code/configs/watermarking_finetuning_freezing.yaml)
    * [watermarking_retraining.yaml](./code/configs/watermarking_retraining.yaml)

    These config files contains the by default parameters to be called in the each of python files. You can edit the parameters in these files only to be directly called in main python files.
    
  #### Below are all the primary code files (with all the descriptions already provided in each and every file, even the order of the execution is also provided with the filename.)

  Each of these files can be run as follows (for eg:) -:

  ```
  python watermarking_finetuning.py
  ```
  
  * [app.py](./code/app.py) - for the end demo purposes.
  * [fgsm_attack.py](./code/fgsm_attack.py) - 2nd but use the below "frontier-stitching" one to generate watermark samples (this file is mainly based on original paper).
  * [frontier-stitching.py](./code/frontier-stitching.py) - 2nd as we have to generate the watermark set.
  * [models.py](./code/models.py) - contains the implenmentation of all models.
  * [model_finepruning.py](./code/model_finepruning.py) - after 4th step (have to performed on the attacker model only.)
  * [model_pruning.py](./code/model_pruning.py) - after 4th step (have to performed on the attacker model only.)
  * [real_model_stealing.py](./code/real_model_stealing.py) - 4th (but use the other two and reason is mentioned in the file itself.)
  * [real_model_stealing_watermark_averaging.py](./code/real_model_stealing_watermark_averaging.py) - 3rd, to perform attack and as well as defense (ownership verification at the time of attack, for averaging the results)
  * [real_model_stealing_watermark_single.py](./code/real_model_stealing_watermark_single.py) - 3rd to perform attack and as well as defense (ownership verification at the time of attack, for single result)
  * [train_original.py](./code/train_original.py) - 1st, to train the original model (no watermarking, simple any model training)
  * [visualization.py](./code/visualization.py) - just for the visualization of the end results
  * [watermarking_finetuning.py](./code/watermarking_finetuning.py) - 3rd (if using extended frontier stitching)
  * [watermarking_retraining.py](./code/watermarking_retraining.py) - 3rd (for frontier stitching)


  
#### to show how the data (watermark samples) wiill be stored/generated using [frontier-stitching.py](./code/frontier-stitching.py) file.
* [data/](./data)
  * [fgsm/](./data/fgsm)
    * [cifar10/](./data/fgsm/cifar10)
      * [false/](./data/fgsm/cifar10/false)
        * [fgsm_0.025_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/false/fgsm_0.025_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_1000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/false/fgsm_0.025_1000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_100_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/false/fgsm_0.025_100_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_2500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/false/fgsm_0.025_2500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/false/fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/false/fgsm_0.025_500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
      * [true/](./data/fgsm/cifar10/true)
        * [fgsm_0.025_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/true/fgsm_0.025_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_1000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/true/fgsm_0.025_1000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_100_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/true/fgsm_0.025_100_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_2500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/true/fgsm_0.025_2500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/true/fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
        * [fgsm_0.025_500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz](./data/fgsm/cifar10/true/fgsm_0.025_500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz)
  
