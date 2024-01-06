# DAWN: Dynamic Adversarial Watermarking of Neural Networks

This paper will appear in the Proceedings of ACM Multimedia 2021.

This repo contains code that allows you to reproduce experiments for the watermarking scheme presented in *DAWN: Dynamic Adversarial Watermarking of Neural Networks*.





# Directory Structure and Files Description

* [configurations/](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations) -- this contains all the configuration files
  * [knockoffnet/](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/knockoffnet)
    * [attack_original.yaml](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/knockoffnet/attack_original.yaml)
  * [mapping/](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/mapping)
    * [mapping-cifar-base.ini](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/mapping/mapping-cifar-base.ini)
    * [mapping-cifar-rn34.ini](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/mapping/mapping-cifar-rn34.ini)
    * [mapping-mnist-l5.ini](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/mapping/mapping-mnist-l5.ini)
  * [perfect/](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/perfect)
    * [caltech-to-caltech-ws250-dn121_drp05.ini](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/perfect/caltech-to-caltech-ws250-dn121_drp05.ini)
    * [cifar-to-cifar-ws250-base.ini](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/perfect/cifar-to-cifar-ws250-base.ini)
  * [pruning.yaml](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/pruning.yaml)
  * [watermark_set.yaml](./dawndynamicadversarialwatermarkingofneuralnetworks/configurations/watermark_set.yaml)

  These config files contains the by default parameters to be called in the each of python files. You can edit the parameters in these files only to be directly called in main python files.
    
#### Below are all the primary code files (with all the descriptions already provided in each and every file)

* [scripts/](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts)

   
     
  * [app.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/app.py)
  * [environment.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/environment.py) - to setup environment for model
  * [experiment.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/experiment.py) - perform training and testing of models.
  * [filter.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/filter.py) - main logic of DAWN to decide to watermark queries.
  * [fineprune.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/fineprune.py) - have to performed on the attacker model only.
  * [main.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/main.py) - for the perfect attacker (Important in the case of perfect attacker only (but this work is focuses on real stealing))
  * [mapping.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/mapping.py) - watermark removal attack specially for DAWN
  * [models.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/models.py)
  * [prune.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/prune.py) - have to performed on the attacker model only.
  * [real_model_stealing_averaging.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/real_model_stealing_averaging.py) - knockoffnet attack with defense evaluation for single/multiple (5) runs (multiple runs to have better understanding of results.)
  * [score.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/score.py)
  * [test_set.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/test_set.py) - to create the test set to perofrm the attack (mainly for the simplicity)
  * [victim_train_attacker_fullquery_retrain.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/victim_train_attacker_fullquery_retrain.py)
  * [visualization.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/visualization.py)
  * [visualization_2.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/visualization_2.py)
  * [watermark_set.py](./dawndynamicadversarialwatermarkingofneuralnetworks/scripts/watermark_set.py) - to create the watermark set such that don't have to run it everytime, saving it once with the particular seed.



### Real Model Extraction attacks along with Defense.
We performed the model extraction attacks "KNOCKOFFNET" and can be run using `real_model_stealing.py` script in the main function modify the configuration path.

```
python scripts/real_model_stealing_averaging.py 
```

<span style="color: green"> Note: DAWN is a inference black-box defense, hence to watermark fraction of queries at the runtime which is to be performed at the attack time itself. Hence, we modify the art-tool box file itself to incorporte the following functionality. The file ```art/attacks/extraction/knockoff_nets.py``` has to be modified for the following functions as shown below: </span>



To run the real stealing attack with verification you need to have the watermark set (as for the demonstration, watermark set is not generated everytime on the fly, but we have saved it for the particular attacker with particular seed.) and which is constructed using

```
python watermark_set.py
```

### Mapping Function
This function is used to prove the Indistinguishability means watermarks dataset should not be removed by the adversary (If A receives different predictions for 𝑥 and 𝑥 + 𝛿 for a small 𝛿, it can discard both 𝑥 and 𝑥 + 𝛿 from its training). To avoid (verify) that we are using mapping.

Experiments with the mapping function can be run using `mapping.py` script in the main function modify the configuration path for which you want to perform the indistingushability test.
```
 python scripts/mapping.py
```




### Perfect Attacker

To run experiments that simulate attacker with perfect knowledge about the victim model, use `main.py`:


```
python scripts/main.py
```



<!-- ### Watermark Removal

#### Detection

Evaluation of the detection method is self contained in `watermark_detection.ipynb`.
It covers MNIST, CIFAR10 and a ResNet variant of CIFAR10.
We provide data from the logit layers (ground truth and watermarks) to conduct these experiments.

#### Noisy Verification

To run the experiments that evaluate resilience of the verification process to added perturbation use the `noisy_verification.py` script.
We provide two attacker/surrogate models obtained using PRADA attacks as well as their corresponding watermarks.

```
usage: noisy_verification.py [-h] [--config_file CONFIG_FILE]
                             [--watermark WATERMARK] [--model MODEL]

optional arguments:
  -h, --help                    show this help message and exit
  --config_file CONFIG_FILE     Configuration file for the experiment.
  --watermark WATERMARK         Path to the saved watermark Loader.
  --model MODEL                 Path to the saved model.
```

CIFAR10:

```
python3 noisy_verification.py \
--config_file configurations/noisy-verification/verification-cifar-prada-single-1000.ini \
--watermark data/scores/attacker_cifar_prada_l5_single_1000_watermark.pkl \
--model data/models/attacker_cifar_prada.pt
```

MNIST:

```
python3 noisy_verification.py \
--config_file configurations/noisy-verification/verification-mnist-prada-single-10.ini \
--watermark data/scores/attacker_mnist_prada_l5_single_10_watermark.pkl \
--model data/models/attacker_mnist_prada.pt
```

Results will printed in the terminal and saved in `data/scores/verification`.

#### Pruning

To run the experiments that evaluate watermark's resilience to pruning use the `prune.py` script.
We provide two attacker/surrogate models obtained using PRADA attacks as well as their corresponding watermarks.

```
usage: prune.py [-h] [--config_file CONFIG_FILE] [--watermark WATERMARK]
                [--model MODEL]

optional arguments:
  -h, --help                    show this help message and exit
  --config_file CONFIG_FILE     Configuration file for the experiment.
  --watermark WATERMARK         Path to the saved watermark Loader.
  --model MODEL                 Path to the saved model.
```

CIFAR10:

```
python3 prune.py \
--config_file configurations/pruning/pruning-mnist-prada-single-10.ini \
--watermark data/scores/attacker_mnist_prada_l5_single_10_watermark.pkl \
--model data/models/attacker_mnist_prada.pt
```

MNIST:

```
python3 prune.py \
--config_file configurations/pruning/pruning-cifar-prada-single-1000.ini \
--watermark data/scores/attacker_cifar_prada_l5_single_1000_watermark.pkl \
--model data/models/attacker_cifar_prada.pt
```

Results will printed in the terminal and saved in `data/scores/pruning`. -->


