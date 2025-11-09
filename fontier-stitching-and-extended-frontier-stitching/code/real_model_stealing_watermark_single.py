"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation for performing model stealing attack on the defended model trained using "watermarking_finetuning.py" file
    This file will perform the attack as well as the ownership verification by checking the presence of the watermark set information (accuracy)
    The difference between this file and the real_model_stealing_watermark_averaging.py file is that it runs for only single run, but to aseess the through results, we have performed the averaging as well.
    These two files can be merged together but for the sake of simplicity we have the following.

"""

import argparse
import warnings
import logging
import sys

warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model

import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from art.attacks.extraction import KnockoffNets
import models
from utils.data_utils import DataManager
from utils.watermark_verifier import WatermarkVerifier
from utils.watermark_metrics import WatermarkMetrics
from utils.experiment_logger import ExperimentLogger, log_reproducibility_info
from datetime import datetime
import time

now = datetime.now().strftime("%d-%m-%Y")

# Force stdout to be unbuffered for immediate output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for gpu in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except AttributeError:
                # Metal GPU doesn't support memory_growth, which is fine
                pass
        print(f"✅ GPU detected: {len(physical_devices)} GPU(s) available", flush=True)
        print(f"   Using: {physical_devices[0].name}", flush=True)
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}", flush=True)
        print("   Falling back to CPU", flush=True)
else:
    print("ℹ️  No GPU detected, using CPU", flush=True)

# Note: ART library may require eager execution disabled for some operations
# Uncomment if needed: tf.compat.v1.disable_eager_execution()


# tf.compat.v1.enable_eager_execution()

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Data preprocessing is now handled by utils.data_utils.DataManager


def model_extraction_attack(dataset_name, adv_data_path_numpy, attacker_model_architecture, number_of_queries,
                            num_epochs_to_steal, dropout, optimizer="adam", lr=0.001, weight_decay=0.00,
                            model_to_attack_path='../models/mnist_original_cnn_epochs_25.h5', results_path=None):
    """
            Main idea
            --------
            Performing the attack and then also veryfying the ownership of the victim model by means watermarkset accuracy.
            If the attacker acheives a good watermark accuracy then victim model can claim that model is stolen from my his/her model.

            Args:
                dataset_name: name of the dataset
                adv_data_path_numpy: watermarkset path
                attacker_model_architecture: architecture which attacker chooses
                number_of_queries: stealing dataset size
                num_epochs_to_steal: number of epochs
                dropout: dropout for the model
                optimizer: optimizer of the model, but anyways we are using by default "Adam".
                lr: learning rate for the model
                weight_decay: if you want to use the weight decay
                model_to_attack_path: victim model path which is already trained with watermarkset.
    """

    print("=" * 60, flush=True)
    print(f"🎯 Starting Model Extraction Attack", flush=True)
    print(f"   Dataset: {dataset_name}", flush=True)
    print(f"   Victim Model: {model_to_attack_path}", flush=True)
    print(f"   Adversarial Data: {adv_data_path_numpy}", flush=True)
    print(f"   Attacker Architecture: {attacker_model_architecture}", flush=True)
    print(f"   Epochs to Steal: {num_epochs_to_steal}", flush=True)
    print("=" * 60, flush=True)
    logger.info(f"Starting model extraction attack on {model_to_attack_path}")
    
    # Set RESULTS_PATH if not provided
    from datetime import datetime
    if results_path is None:
        now = datetime.now().strftime("%d-%m-%Y")
        RESULTS_PATH = f"../results/attack_finetuned{now}"
    else:
        RESULTS_PATH = results_path
    
    # Set other paths (needed for file operations)
    LOSS_Acc_FOLDER = "losses_acc"
    now = datetime.now().strftime("%d-%m-%Y")
    MODEL_PATH = f"../models/attack_finetuned{now}"
    DATA_PATH = "../data"
    
    # Extract which_adv from adversarial path
    which_adv = adv_data_path_numpy.replace("\\", "/").split("/")[-2] if "/" in adv_data_path_numpy.replace("\\", "/") else "true"
    
    # Create directories if they don't exist
    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, which_adv), exist_ok=True)
    
    if not os.path.exists(os.path.join(MODEL_PATH, which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, which_adv), exist_ok=True)
    
    # Initialize ExperimentLogger for comprehensive logging
    exp_logger = ExperimentLogger("model_extraction_attack", output_dir=RESULTS_PATH)
    
    # Log reproducibility info
    log_reproducibility_info(output_dir=str(exp_logger.experiment_dir), seed=0)
    
    # Log hyperparameters
    exp_logger.log_hyperparameters(
        dataset_name=dataset_name,
        attacker_model_architecture=attacker_model_architecture,
        num_epochs_to_steal=num_epochs_to_steal,
        dropout=dropout,
        optimizer=str(optimizer),
        lr=lr,
        weight_decay=weight_decay,
        query_budgets=number_of_queries,
        victim_model_path=model_to_attack_path,
        adv_data_path=adv_data_path_numpy
    )

    # Use centralized DataManager for data loading
    x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape = DataManager.load_and_preprocess_with_adversarial(
        dataset_name=dataset_name, 
        adv_data_path=adv_data_path_numpy
    )
    
    # Get num_classes from dataset config
    num_classes = DataManager.get_dataset_info(dataset_name)['num_classes']

    models_mapping = {"mnist_l2": models.MNIST_L2, "cifar10_base_2": models.CIFAR10_BASE_2, "resnet34": models.ResNet34,
                      "cifar10_wideresnet": models.wide_residual_network}
    num_epochs = num_epochs_to_steal
    
    log_file_path = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                              "_".join((dataset_name, str(num_epochs),
                                        model_to_attack_path.replace("\\", "/").split("/")[-2] + "_logs.txt")))
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file1 = open(log_file_path, "w")

    print(f"📦 Loading victim model from: {model_to_attack_path}", flush=True)
    logger.info(f"Loading victim model from: {model_to_attack_path}")
    model = load_model(model_to_attack_path, compile=False)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer="adam", metrics=['accuracy'])
    print("✅ Victim model loaded successfully", flush=True)
    
    # Log victim model size
    victim_total_params = model.count_params()
    victim_model_size_mb = victim_total_params * 4 / (1024 * 1024)
    print(f"📊 Victim model size: {victim_total_params:,} parameters ({victim_model_size_mb:.2f} MB)")
    exp_logger.metrics['victim_model_size_params'] = int(victim_total_params)
    exp_logger.metrics['victim_model_size_mb'] = float(victim_model_size_mb)

    ## Evaluating the victim model accuracy on the watermark set.
    print("🔍 Evaluating victim model on watermark set...", flush=True)
    logger.info("Evaluating victim model on watermark set")
    victim_watermark_acc = model.evaluate(x_adv, y_adv, verbose=0)[1]
    print(f"   Victim model watermark accuracy: {victim_watermark_acc:.4f}", flush=True)
    logger.info(f"Victim model watermark accuracy: {victim_watermark_acc:.4f}")
    file1.write("Just After loading victim model adv acc is: " + str(victim_watermark_acc) + "\n")
    exp_logger.metrics['victim_watermark_acc'] = float(victim_watermark_acc)

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    if attacker_model_architecture == "resnet34":

        classifier_original = TensorFlowV2Classifier(model, nb_classes=num_classes,
                                                     input_shape=(x_train[1], x_train[2], x_train[3]))

    else:

        classifier_original = KerasClassifier(model, clip_values=(0, 1), use_logits=False)

    im_shape = x_train[0].shape

    results = []
    results_adv = []

    print(f"\n🔄 Starting attacks with query budgets: {number_of_queries}", flush=True)
    logger.info(f"Starting attacks with query budgets: {number_of_queries}")
    total_queries = len(number_of_queries)
    current_query = 0

    ## performing the attack according to the query budget.
    for len_steal in number_of_queries:
        current_query += 1
        print("", flush=True)
        print("-" * 60, flush=True)
        print(f"📊 Attack {current_query}/{total_queries}: Query budget = {len_steal}", flush=True)
        print("-" * 60, flush=True)
        logger.info(f"Starting attack with query budget: {len_steal}")
        
        indices = np.random.permutation(len(x_test))
        x_steal = x_test[indices[:len_steal]]
        y_steal = y_test[indices[:len_steal]]
        x_test0 = x_test[indices[len_steal:]]
        y_test0 = y_test[indices[len_steal:]]
        
        print(f"   Stealing dataset size: {len(x_steal)} samples", flush=True)

        attack_catalogue = {"KnockoffNet": KnockoffNets(classifier=classifier_original,
                                                        batch_size_fit=64,
                                                        batch_size_query=64,
                                                        nb_epochs=num_epochs,
                                                        nb_stolen=len_steal,
                                                        use_probability=False)}

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        def train_step(model1, images, labels):

            with tf.GradientTape() as tape:
                prediction = model1(images)
                loss = loss_object(labels, prediction)
                file1.write(f"\n Loss of attacker model: {loss:.3f}")
                file1.write("\n")
                # print("loss", loss)

            grads = tape.gradient(loss, model1.trainable_weights)
            optimizer.apply_gradients(zip(grads, model1.trainable_weights))

        for name, attack in attack_catalogue.items():
            print(f"   🏗️  Creating attacker model: {attacker_model_architecture}", flush=True)
            logger.info(f"Creating attacker model: {attacker_model_architecture}")

            ## setting up the attacker model.
            if attacker_model_architecture == "resnet34":
                model_name, model_stolen = models_mapping[attacker_model_architecture]().call(input_shape)

                # model_stolen.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer="adam", metrics=['accuracy'])

            else:
                if dropout:
                    model_name, model_stolen = models_mapping[attacker_model_architecture](dropout=dropout, num_classes=num_classes)
                else:
                    model_name, model_stolen = models_mapping[attacker_model_architecture](num_classes=num_classes)

                model_stolen.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer="adam", metrics=['accuracy'])

            print(f"   ✅ Attacker model created: {model_name}", flush=True)

            if attacker_model_architecture == "resnet34":

                classifier_stolen = TensorFlowV2Classifier(model_stolen, nb_classes=num_classes, loss_object=loss_object,
                                                           input_shape=input_shape, channels_first=False,
                                                           train_step=train_step)

            else:

                classifier_stolen = KerasClassifier(model_stolen, clip_values=(0, 1), use_logits=False)

            ## performing the attack.
            print(f"   ⚔️  Performing model extraction attack...", flush=True)
            print(f"      This may take a few minutes (training {num_epochs} epochs)...", flush=True)
            logger.info(f"Performing model extraction attack with {len_steal} queries, {num_epochs} epochs")
            
            # Track attack time
            attack_start_time = time.time()
            classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)
            attack_time = time.time() - attack_start_time
            print(f"   ✅ Attack completed in {attack_time:.2f} seconds", flush=True)

            ## evaluating the attacked model on test set
            print(f"   📊 Evaluating stolen model on test set...", flush=True)
            acc = classifier_stolen.model.evaluate(x_test, y_test, verbose=0)[1]
            print(f"   ✅ Test accuracy with {len_steal} queries: {acc:.4f}", flush=True)
            logger.info(f"Test accuracy with {len_steal} queries: {acc:.4f}")
            file1.write(f"Victim model {model_to_attack_path}")
            file1.write(f"test acc with {len_steal} is {acc}\n")
            results.append((name, len_steal, acc))

            # test with adversarial data
            # evaluating the attacked model on adversarial set/watermark set.
            print(f"   🔍 Evaluating stolen model on watermark set...", flush=True)
            acc_adv = classifier_stolen.model.evaluate(x_adv, y_adv, verbose=0)[1]
            print(f"   ✅ Watermark accuracy with {len_steal} queries: {acc_adv:.4f}", flush=True)
            logger.info(f"Watermark accuracy with {len_steal} queries: {acc_adv:.4f}")
            file1.write(f"adv acc with {len_steal} is {acc_adv}\n")
            results_adv.append((name, len_steal, acc_adv))

            # Statistical theft verification
            print(f"   🔍 Verifying theft using statistical methods...", flush=True)
            verifier = WatermarkVerifier(
                victim_acc=victim_watermark_acc,  # Use victim's watermark accuracy
                num_classes=num_classes,
                watermark_size=len(x_adv)
            )
            verification_result = verifier.verify_theft(
                suspected_acc=acc_adv,  # Stolen model's watermark accuracy
                threshold_ratio=0.5,
                confidence=0.99
            )
            print(f"   📊 Theft Verification Results:", flush=True)
            print(f"      Is stolen: {verification_result['is_stolen']}", flush=True)
            print(f"      Confidence: {verification_result['confidence']:.4f}", flush=True)
            print(f"      P-value: {verification_result['p_value']:.6f}", flush=True)
            print(f"      Threshold: {verification_result['threshold']:.4f}", flush=True)
            logger.info(f"Theft verification: is_stolen={verification_result['is_stolen']}, "
                       f"confidence={verification_result['confidence']:.4f}, "
                       f"p_value={verification_result['p_value']:.6f}")
            file1.write(f"Theft verification with {len_steal} queries: "
                       f"is_stolen={verification_result['is_stolen']}, "
                       f"confidence={verification_result['confidence']:.4f}, "
                       f"p_value={verification_result['p_value']:.6f}\n")

            # Calculate comprehensive metrics
            print(f"   📈 Calculating comprehensive watermark metrics...", flush=True)
            metrics = WatermarkMetrics.calculate_all_metrics(
                victim_model=model,
                stolen_model=classifier_stolen.model,
                x_test=x_test0,
                y_test=y_test0,
                x_watermark=x_adv,
                y_watermark=y_adv,
                batch_size=128
            )
            print(f"   📊 Comprehensive Metrics:", flush=True)
            print(f"      Fidelity: {metrics['fidelity']:.4f}", flush=True)
            print(f"      Watermark Retention: {metrics['watermark_retention']:.4f} ({metrics['watermark_retention']*100:.2f}%)", flush=True)
            print(f"      Test Accuracy Gap: {metrics['test_acc_gap']:.4f}", flush=True)
            print(f"      KL Divergence: {metrics['kl_divergence']:.4f}", flush=True)
            print(f"      Detectability Score: {metrics['detectability']:.4f}", flush=True)
            logger.info(f"Comprehensive metrics: fidelity={metrics['fidelity']:.4f}, "
                       f"retention={metrics['watermark_retention']:.4f}, "
                       f"detectability={metrics['detectability']:.4f}")
            file1.write(f"Comprehensive metrics with {len_steal} queries: "
                       f"fidelity={metrics['fidelity']:.4f}, "
                       f"retention={metrics['watermark_retention']:.4f}, "
                       f"test_acc_gap={metrics['test_acc_gap']:.4f}, "
                       f"kl_divergence={metrics['kl_divergence']:.4f}, "
                       f"detectability={metrics['detectability']:.4f}\n")
            
            # Log stolen model size
            stolen_total_params = classifier_stolen.model.count_params()
            stolen_model_size_mb = stolen_total_params * 4 / (1024 * 1024)
            
            # Log attack results to ExperimentLogger
            exp_logger.log_attack_results(
                query_budget=len_steal,
                test_acc=acc,
                watermark_acc=acc_adv,
                verification_result=verification_result,
                comprehensive_metrics=metrics,
                attack_time=attack_time,
                stolen_model_size_params=stolen_total_params,
                stolen_model_size_mb=stolen_model_size_mb
            )

            # Save model in Keras format (.keras) instead of legacy HDF5 (.h5)
            model_save_path = os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                             "_".join((dataset_name, str(len_steal), str(num_epochs),
                                       adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                           0] + ".keras")))
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            print(f"   💾 Saving stolen model to: {model_save_path}", flush=True)
            classifier_stolen.model.save(model_save_path)
            print(f"   ✅ Model saved successfully", flush=True)

    print(f"\n📈 Generating visualization...", flush=True)
    logger.info("Generating visualization")
    
    # Save plot to both old location (for compatibility) and ExperimentLogger plots directory
    image_save_name_old = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                   "_".join((dataset_name, str(num_epochs),
                                             model_to_attack_path.replace("\\", "/").split("/")[
                                                 -2] + "TestandWatermarkAcc.png")))
    os.makedirs(os.path.dirname(image_save_name_old), exist_ok=True)
    
    # Also save to ExperimentLogger plots directory
    image_save_name_new = str(exp_logger.experiment_dir / "plots" / "test_and_watermark_accuracy.png")
    os.makedirs(os.path.dirname(image_save_name_new), exist_ok=True)

    df = pd.DataFrame(results, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, group in df.groupby("Method Name"):
        group.plot(1, 2, ax=ax, label="Test acc", linestyle='--', marker='o', color='tab:purple')

    df_adv = pd.DataFrame(results_adv, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
    ax.set_xlabel("Stealing Dataset Size")
    ax.set_ylabel("Stolen Model Test and Adversarial Accuracy")
    for name, group in df_adv.groupby("Method Name"):
        group.plot(1, 2, ax=ax, label="Watermark acc", linestyle='--', marker='o', color='tab:orange')
    
    # Save to both locations
    plt.savefig(image_save_name_old)
    plt.savefig(image_save_name_new)
    plt.close()  # Close figure to free memory
    print(f"   ✅ Visualization saved to: {image_save_name_old}", flush=True)
    print(f"   ✅ Visualization saved to: {image_save_name_new}", flush=True)
    logger.info(f"Visualization saved to: {image_save_name_old} and {image_save_name_new}")
    
    # Save DataFrames to CSV for research paper analysis
    csv_test_path = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                 "_".join((dataset_name, str(num_epochs),
                                           model_to_attack_path.replace("\\", "/").split("/")[-2] + "_test_accuracy.csv")))
    csv_watermark_path = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                      "_".join((dataset_name, str(num_epochs),
                                                model_to_attack_path.replace("\\", "/").split("/")[-2] + "_watermark_accuracy.csv")))
    
    df.to_csv(csv_test_path, index=False)
    df_adv.to_csv(csv_watermark_path, index=False)
    print(f"   ✅ CSV files saved:", flush=True)
    print(f"      Test accuracy: {csv_test_path}", flush=True)
    print(f"      Watermark accuracy: {csv_watermark_path}", flush=True)
    logger.info(f"CSV files saved: {csv_test_path}, {csv_watermark_path}")
    
    file1.close()
    print(f"   ✅ Log file saved", flush=True)
    
    # Save all ExperimentLogger data
    exp_logger.save_all()
    
    # Create LaTeX table from results
    latex_table = exp_logger.create_latex_table("attack_results.tex")
    if latex_table:
        print(f"   ✅ LaTeX table saved to: {exp_logger.experiment_dir / 'tables' / 'attack_results.tex'}", flush=True)
    
    print(f"   ✅ Comprehensive experiment data saved to: {exp_logger.experiment_dir}", flush=True)
    
    print("=" * 60, flush=True)
    print(f"✅ Model extraction attack completed successfully!", flush=True)
    print("=" * 60, flush=True)
    logger.info("Model extraction attack completed successfully")

    return df, df_adv


if __name__ == "__main__":
    # Optimized configuration values for model extraction attack
    dataset_name = "cifar10"
    epochs_extract = 50
    attacker_model_architecture = "cifar10_base_2"
    optimizer = "adam"
    lr = 0.001  # Optimized from 0.01 for better attack training
    weight_decay = 0
    dropout = 0
    which_adv = "true"

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    RESULTS_PATH = f"../results/attack_finetuned{now}"
    LOSS_Acc_FOLDER = "losses_acc"
    MODEL_PATH = f"../models/attack_finetuned{now}"
    DATA_PATH = "../data"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, which_adv))

    if not os.path.exists(os.path.join(MODEL_PATH, which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, which_adv))

    params = {"dataset_name": dataset_name,
              "attacker_model_architecture": attacker_model_architecture, "optimizer": optimizer,
              "dropout": dropout, "lr": lr, "weight_decay": weight_decay,
              "epochs_extract": epochs_extract}

            ## ------------------------------- IMPORTANT ---------------------------##
            ## to remove the overhead present in the real_model_stealing.py file for the mnually changing things in the loop.
            ## here we improved it by choosing the finetuned victim model path and the corresponding adversioal path so that one can verify the watermark (adversarial) accuracy of the attacker model
            ## -----------------------------------------------------------------------------##

        # adv_file_path = ["../data/fgsm/mnist/fgsm_0.5_250_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz",
        #                  "../data/fgsm/mnist/fgsm_0.5_2500_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz",
        #                  "../data/fgsm/mnist/fgsm_0.25_500_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz",
        #                  "../data/fgsm/mnist/fgsm_0.25_2500_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz"]
        # finetuned_model_path = [
        #     "../models/finetuned_retraining/mnist_100_MNIST_l20.2fgsm_0.5_250_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5",
        #     "..//models/finetuned_retraining/mnist_100_MNIST_l20.2fgsm_0.5_2500_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5",
        #     "../models/finetuned_retraining_24-08-2023/mnist_100_MNIST_l20.0fgsm_0.25_500_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5",
        #     "../models/finetuned_retraining_24-08-2023/mnist_100_MNIST_l20.0fgsm_0.25_2500_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5"]

    # adv_file_path = ["../data/fgsm/mnist/true/fgsm_0.1_10000_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz"]

    # finetuned_model_path = [
    #     "../models/finetuned_finetuning_11-09-2023/true/final_mnist_25_25_mnist_20_MNIST_l20.0fgsm_0.1_10000_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_final.h5"]

    # finetuned_model_path = [
    #     "../models/finetuned_retraining_14-09-2023/true/cifar10_100_CIFAR10_BASE_2fgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5"]

    # adv_file_path = ["../data/fgsm/mnist/true/fgsm_0.4_1000_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz"]

    # finetuned_model_path = [
    #    "../models/finetuned_retraining_28-08-2023/true/mnist_100_MNIST_l20.0fgsm_0.4_1000_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5"]

    # adv_file_path = ["../data/fgsm/cifar10resnet_255_preprocess/true/fgsm_0.035_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best.npz"]
    # finetuned_model_path = [
    #     "../models/finetuned_finetuning_02-09-2023/true/cifar10_25_25_bestfgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5"]

    # dataset_name = "mnist"
    # adv_file_path = ["../data/fgsm/cifar10/fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz","../data/fgsm/cifar10/fgsm_0.1_2500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz",
    #                  "../data/fgsm/cifar10/fgsm_0.1_1000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz", "../data/fgsm/cifar10/fgsm_0.1_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"]
    # finetuned_model_path = ["../models/finetuned_finetuning_20-08-2023/cifar10_100_bestfgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5","../models/finetuned_finetuning_20-08-2023/cifar10_100_bestfgsm_0.1_2500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5",
    #                         "../models/finetuned_finetuning_20-08-2023/cifar10_100_bestfgsm_0.1_1000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5", "../models/finetuned_finetuning_20-08-2023/cifar10_100_bestfgsm_0.1_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5"]

    # adv_file_path = ["../data/fgsm/cifar10/fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"]

    # finetuned_model_path = ["../models/finetuned_retraining_21-08-2023/cifar10_100_CIFAR10_BASE_2fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5"]

    # adv_file_path = [
    #     "../data/fgsm/cifar10/true/fgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"]

    # finetuned_model_path = ["../models/finetuned_retraining_22-08-2023/cifar10_100_CIFAR10_BASE_2fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5"]

        # finetuned_model_path = ["../models/finetuned_finetuning_06-09-2023/true/mnist_25_25_mnist_20_MNIST_l20.0fgsm_0.25_10000_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5"]

        # adv_file_path = ["../data/fgsm/mnist/true/fgsm_0.25_10000_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz"]

    # adv_file_path = [
    #     "../data/fgsm/cifar10resnet_255_preprocess/true/fgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best.npz"]

    # finetuned_model_path = [
    #     "../models/finetuned_finetuning_08-09-2023/true/final_cifar10resnet_255_preprocess_10_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best/Victim_checkpoint_final.h5"]

    # adv_file_path = ["../data/fgsm/cifar10resnet_255_preprocess/true/fgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best.npz"]

    # finetuned_model_path = ["../models/finetuned_finetuning_08-09-2023/true/final_cifar10resnet_255_preprocess_10_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best/Victim_checkpoint_final.h5"]


    print("=" * 60, flush=True)
    print("🚀 Starting Model Stealing Attack with Watermark Verification", flush=True)
    print("=" * 60, flush=True)
    print(f"📋 Configuration:", flush=True)
    print(f"   Dataset: {dataset_name}", flush=True)
    print(f"   Attacker Architecture: {attacker_model_architecture}", flush=True)
    print(f"   Epochs to Extract: {epochs_extract}", flush=True)
    print(f"   Query Budgets: [250, 500, 1000, 5000, 10000, 20000]", flush=True)
    print("", flush=True)
    logger.info("Starting model stealing attack with watermark verification")

    # Paths to adversarial examples and watermarked models
    # Update these paths to match your generated files
    adv_file_paths = [
        "../data/fgsm/cifar10/true/fgsm_0.01_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz",
        "../data/fgsm/cifar10/true/fgsm_0.015_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"
    ]

    finetuned_model_paths = [
        "../models/finetuned_finetuning_09-11-2025/true/cifar10_10_10_cifar10_30_CIFAR10_BASE_2fgsm_0.01_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.keras",
        "../models/finetuned_finetuning_09-11-2025/true/cifar10_10_10_cifar10_30_CIFAR10_BASE_2fgsm_0.015_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.keras"
    ]

    total_models = len(adv_file_paths)
    current_model = 0

    for adv_file_path, model_path in zip(adv_file_paths, finetuned_model_paths):
        current_model += 1
        print("", flush=True)
        print("=" * 60, flush=True)
        print(f"🔄 Processing Model {current_model}/{total_models}", flush=True)
        print("=" * 60, flush=True)
        
        # Validate paths exist
        if not os.path.exists(adv_file_path):
            print(f"❌ ERROR: Adversarial file not found: {adv_file_path}", flush=True)
            logger.error(f"Adversarial file not found: {adv_file_path}")
            continue
            
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model file not found: {model_path}", flush=True)
            logger.error(f"Model file not found: {model_path}")
            continue
        
        adv_data_path_numpy = adv_file_path
        model_to_attack_path = model_path

        final_df_test = pd.DataFrame(columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
        final_df_adv = pd.DataFrame(columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))

        df, df_adv = model_extraction_attack(dataset_name, adv_data_path_numpy,
                                                    attacker_model_architecture,
                                                    number_of_queries=[250, 500, 1000, 5000, 10000, 20000],
                                                    num_epochs_to_steal=epochs_extract, dropout=dropout,
                                                    optimizer=optimizer,
                                                    lr=lr, weight_decay=weight_decay,
                                                    model_to_attack_path=model_to_attack_path)
        
        print(f"✅ Completed processing model {current_model}/{total_models}", flush=True)
    
    print("", flush=True)
    print("=" * 60, flush=True)
    print("✅ All model extraction attacks completed!", flush=True)
    print("=" * 60, flush=True)
    logger.info("All model extraction attacks completed successfully")



