# Watermarking as Defense against Model Extraction Attacks on Deep Learning Models

This repository contains black-box watermarking defense mechanisms for model extraction attacks.

Mainly contains these methods implementation -:

* [DAWN Dynamic Adversarial Watermarking of Neural Networks](https://arxiv.org/pdf/1906.00830.pdf)
* [Entangled watermarking](https://arxiv.org/pdf/2002.12200.pdf)
* [Frontier Stitching](https://arxiv.org/abs/1711.01894) 
* Extended Frontier Stitching - extended version of Frontier Stitching (proposed method of this thesis to handle model extraction via api access by extending frontier stitching idea)

Deatiled results are also presented in thesis document as well as the meeting presentations in [extra folder](extras)


## Installation

- Clone the repository
- Setup the virtual environment. This repo is mainly based on Python 3.8.10 (but tested on Python 3.10.x also - worked completely fine.)
- Already provided the ``requirements.txt`` file to setup the common requirements used throughout this code base. (Incase some library version is not supported, it is easy to use with your Python compatible version.)
- `Just install the requirements using ``pip install -r requirements.txt``

## Directory structure

### This contains the directory structure and the each of the folder represents to the each of the above mentioned defense technique. Extended frontier stitching code is also included in frontier-stitching.

### <font color="green"> Sub READMEs are also available inside each of these folders. </font> 

* [dawndynamicadversarialwatermarkingofneuralnetworks/](model-extraction-defense/modeldefense/dawndynamicadversarialwatermarkingofneuralnetworks)
  * This folder main code is present in [scripts](model-extraction-defense/modeldefense/dawndynamicadversarialwatermarkingofneuralnetworks/scripts) and also in [utils](model-extraction-defense/modeldefense/dawndynamicadversarialwatermarkingofneuralnetworks/utils)
  * For this folder, make sure to run ``pip install -e .`` before running the scripts because it is nested differently for utils and scripts, so to make sure everything works out, you have to install this as a package itself as mentioned by the command. This is why we have __init__.py file in the directory.
  * This is implemented in ``torch 2.0.0+cu117``
* [entangled-watermark/](model-extraction-defense/modeldefense/entangled-watermark) 
  * This folder main code is present in folder [Tensorflowv2](model-extraction-defense/modeldefense/entangled-watermark/Tensorflowv2)
  * Originally this paper code is present on thier official github repository but unfortunately it was in ``TensorFlow 1.3.x`` which was not very compatibel for the attacks we were performing.
  * Hence have to modify the code to make it to ``TensorFlow 2.10.x``. We also tried to convert this code to PyTorch (which can be refered as the future work.)
* [frontier-stitching-and-extended-frontier-stitching/](model-extraction-defense/modeldefense/fontier-stitching-and-extended-frontier-stitching)
  * This folder main code conatins in folder: [code](model-extraction-defense/modeldefense/fontier-stitching-and-extended-frontier-stitching/code).
  * This is implemented in ``TensiorFlow 2.10.x``
  
### Common Remark
- Each of this sub folders contains the configurations files, since we have set the varaibles which user has to pass through configuration file.
- For each of this can be found in different subfolders may be (this can be considered as future work! - to have the same directory structure for each of these defense techniques)
- Though the details of configuration files is already mentioned in the respective ReadMe.md file
- To perform the attack - we have used [art-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- In this work, we focused on KnockoffNets attack which is one of the strongest attacks on deep learning models.
- For the attack we have made assumption that attacker has knowledge of the dataset for which the victim model is trained with. Reason: This is the strong assumption for the defense, even if the attacker has full knowledge of the training data but not of the victim model, we can claim the ownership if it is stolen.
- Each of the above mentioned directories contains the  filename derived with ``real_model_stealing.py`` which contains the code to perform the attack on the watermarked model (watermarked victim model) and also during attack perfroming the verification on the watermark set.


## This is the demo for extended frontier stitching.

View Demo : [Video](streamlit-app-2023-12-15-01-12-99.webm)

### App code is available in [file](model-extraction-defense/modeldefense/fontier-stitching-and-extended-frontier-stitching/code/app.py) - to understand more how it is working in backend.

To run the code or app locally run ```streamlit run app.py``` from that [folder](model-extraction-defense/modeldefense/fontier-stitching-and-extended-frontier-stitching/code) 




## Contact

Incase of any issues on this repository please write at : garg.ridhima72@gmail.com 