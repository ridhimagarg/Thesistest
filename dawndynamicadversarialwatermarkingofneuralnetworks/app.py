"""
@author: Ridhima Garg

Introduction:
    This file contains the code of the web app to load the model and run the ownership verification.

"""

import warnings

import cv2
import streamlit as st

warnings.filterwarnings('ignore')

import os
import numpy as np
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.title('Defense from Model Stealing Attacks!')
st.subheader('Claiming Model Ownership')
labels_mapping = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

## Given an option to select the mdoel.
option = st.selectbox("Select the model", ("cifar10_low_capacity", "cifar10_high_capacity"))

## if the option is selected then just load that model.
if option == "cifar10_low_capacity":
    numpy_arr_path = "../data/fgsm/cifar10/true/samples_fgsm_0.035_10000/"

    numpy_arr = np.load(
        "../data/fgsm/cifar10/true/fgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz")

    # if st.button("Generated Prediction Model without Defense"):
    original_model = keras.models.load_model(
        "../models/original_20-08-2023/cifar10_30_CIFAR10_BASE_2/Original_checkpoint_best.h5", compile=False)
    finetuned_model = keras.models.load_model(
        "../models/finetuned_finetuning_02-09-2023/true/cifar10_25_25_bestfgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5",
        compile=False)
    attack_finetuned_model = keras.models.load_model(
        "../models/attack_finetuned02-09-2023/true/cifar10_20000_50_fgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.h5",
        compile=False)

if option == "cifar10_high_capacity":
    numpy_arr_path = "../data/fgsm/cifar10resnet_255_preprocess/true/samples_fgsm_0.025_10000/"

    numpy_arr = np.load(
        "../data/fgsm/cifar10resnet_255_preprocess/true/fgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best.npz")

    # if st.button("Generated Prediction Model without Defense"):
    original_model = keras.models.load_model(
        "../models/original_07-09-2023/cifar10_250_WideResNet_255_preprocess/Original_checkpoint_best.h5",
        compile=False)
    finetuned_model = keras.models.load_model(
        "../models/finetuned_finetuning_08-09-2023/true/final_cifar10resnet_255_preprocess_10_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best/Victim_checkpoint_final.h5",
        compile=False)
    attack_finetuned_model = keras.models.load_model(
        "../models/attack_finetuned12-09-2023/true/cifar10resnet_255_preprocess_10000_50_fgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best.h5",
        compile=False)

## One has to also choose the particular watermark samples files to test the ownership verification.
uploaded_files = st.file_uploader("Choose a file (watermark sample) to claim the ownership of your model",
                                  accept_multiple_files=True)

## Performing the prediction of the 3 different models (Unwatermarked, watermarked, attacked) on the watermarked samples files
if uploaded_files is not None:
    # To read file as bytes:
    for uploaded_file in uploaded_files:
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.resize(image, (150, 150))
        st.image(image, channels="BGR")

        numpy_arr = np.load(numpy_arr_path + uploaded_file.name.split(".")[0] + ".npz")
        print(uploaded_file.name)
        print(numpy_arr["arr_0"].shape)
        prediction = original_model.predict(np.expand_dims(numpy_arr['arr_0'], axis=0))
        st.info(f"Model (without watermark) Prediction: **{labels_mapping[np.argmax(prediction)]}**")

        prediction = finetuned_model.predict(np.expand_dims(numpy_arr['arr_0'], axis=0))
        st.info(f"Model (with watermark finetuned) Prediction: **{labels_mapping[np.argmax(prediction)]}**")

        st.write("Claiming ownership on attacked model")
        prediction = attack_finetuned_model.predict(np.expand_dims(numpy_arr['arr_0'], axis=0))
        st.info(f"Attack Model Prediction: **{labels_mapping[np.argmax(prediction)]}**")























## ---------------------- extras -------------------------##
# st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;")
# st.markdown("***")
# st.subheader('Attacking the original (watermarked) model', divider='rainbow')
#
# def data_preprocessing(dataset_name, adv_data_path_numpy):
#     if dataset_name == 'mnist':
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         img_rows, img_cols, num_channels = 28, 28, 1
#         num_classes = 10
#
#     elif dataset_name == 'cifar10':
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#         img_rows, img_cols, num_channels = 32, 32, 3
#         num_classes = 10
#
#     elif dataset_name == "cifar10resnet":
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#         img_rows, img_cols, num_channels = 32, 32, 3
#         num_classes = 10
#
#     elif dataset_name == "cifar10resnet_255_preprocess":
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#         img_rows, img_cols, num_channels = 32, 32, 3
#         num_classes = 10
#
#     else:
#         raise ValueError('Invalid dataset name')
#
#     idx = np.random.randint(x_train.shape[0], size=len(x_train))
#     x_train = x_train[idx, :]
#     y_train = y_train[idx]
#
#     # specify input dimensions of each image
#     input_shape = (img_rows, img_cols, num_channels)
#
#     # reshape x_train and x_test
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)
#
#     # convert class labels (from digits) to one-hot encoded vectors
#     y_train = keras.utils.to_categorical(y_train, num_classes)
#     y_test = keras.utils.to_categorical(y_test, num_classes)
#
#     # convert int to float
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#
#     # normalise
#     if dataset_name != 'cifar10resnet':
#         x_train /= 255
#         x_test /= 255
#
#     else:
#
#         mean = [125.3, 123.0, 113.9]
#         std = [63.0, 62.1, 66.7]
#
#         for i in range(3):
#             x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
#             x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
#
#     # load adversarial data
#     adv = np.load(adv_data_path_numpy)
#     x_adv, y_adv = adv['arr_1'], adv['arr_2']
#     print(x_adv.shape)
#     print(y_adv.shape)
#
#     return x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape
#
# def model_extraction_attack(dataset_name, adv_data_path_numpy, attacker_model_architecture, number_of_queries,
#                             num_epochs_to_steal, dropout, optimizer="adam", lr=0.001, weight_decay=0.00,
#                             model_to_attack_path='../models/mnist_original_cnn_epochs_25.h5'):
#     x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape = data_preprocessing(dataset_name, adv_data_path_numpy)
#
#     models_mapping = {"mnist_l2": models.MNIST_L2, "cifar10_base_2": models.CIFAR10_BASE_2, "resnet34": models.ResNet34,
#                       "cifar10_wideresnet": models.wide_residual_network}
#     num_epochs = num_epochs_to_steal
#     file1 = open(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
#                               "_".join((dataset_name, str(num_epochs),
#                                         model_to_attack_path.replace("\\", "/").split("/")[-2] + "_logs.txt"))), "w")
#
#     model = load_model(model_to_attack_path, compile=False)
#     model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
#
#     acc_adv = model.evaluate(x_adv, y_adv)[1]
#     print("Just After loading victim model adv acc is:", acc_adv)
#     file1.write("Just After loading victim model adv acc is: " + str(acc_adv) + "\n")
#
#     loss_object = tf.keras.losses.CategoricalCrossentropy()
#
#     if attacker_model_architecture == "resnet34":
#
#         classifier_original = TensorFlowV2Classifier(model, nb_classes=10,
#                                                      input_shape=(x_train[1], x_train[2], x_train[3]))
#
#     else:
#
#         classifier_original = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
#
#     im_shape = x_train[0].shape
#
#     results = []
#     results_adv = []
#
#     for len_steal in number_of_queries:
#         indices = np.random.permutation(len(x_test))
#         x_steal = x_test[indices[:len_steal]]
#         y_steal = y_test[indices[:len_steal]]
#         x_test0 = x_test[indices[len_steal:]]
#         y_test0 = y_test[indices[len_steal:]]
#
#         attack_catalogue = {"KnockoffNet": KnockoffNets(classifier=classifier_original,
#                                                         batch_size_fit=64,
#                                                         batch_size_query=64,
#                                                         nb_epochs=num_epochs,
#                                                         nb_stolen=len_steal,
#                                                         use_probability=False)}
#
#         optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#
#         def train_step(model1, images, labels):
#
#             with tf.GradientTape() as tape:
#                 prediction = model1(images)
#                 loss = loss_object(labels, prediction)
#                 file1.write(f"\n Loss of attacker model: {loss:.3f}")
#                 file1.write("\n")
#                 # print("loss", loss)
#
#             grads = tape.gradient(loss, model1.trainable_weights)
#             optimizer.apply_gradients(zip(grads, model1.trainable_weights))
#
#         for name, attack in attack_catalogue.items():
#
#             if attacker_model_architecture == "resnet34":
#                 model_name, model_stolen = models_mapping[attacker_model_architecture]().call(input_shape)
#
#                 # model_stolen.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
#
#             else:
#                 if dropout:
#                     model_name, model_stolen = models_mapping[attacker_model_architecture](dropout)
#                 else:
#                     model_name, model_stolen = models_mapping[attacker_model_architecture]()
#
#                 model_stolen.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
#
#             if attacker_model_architecture == "resnet34":
#
#                 classifier_stolen = TensorFlowV2Classifier(model_stolen, nb_classes=10, loss_object=loss_object,
#                                                            input_shape=input_shape, channels_first=False,
#                                                            train_step=train_step)
#
#             else:
#
#                 classifier_stolen = KerasClassifier(model_stolen, clip_values=(0, 1), use_logits=False)
#
#             classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)
#             acc = classifier_stolen.model.evaluate(x_test, y_test)[1]
#             print(f"test acc with {len_steal} is {acc}")
#             file1.write(f"Victim model {model_to_attack_path}")
#             file1.write(f"test acc with {len_steal} is {acc}\n")
#             results.append((name, len_steal, acc))
#
#             # test with adversarial data
#             acc_adv = classifier_stolen.model.evaluate(x_adv, y_adv)[1]
#             print(f"adv acc with {len_steal} is {acc_adv}")
#             file1.write(f"adv acc with {len_steal} is {acc_adv}\n")
#             results_adv.append((name, len_steal, acc_adv))
#
#     image_save_name = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
#                                    "_".join((dataset_name, str(num_epochs),
#                                              model_to_attack_path.replace("\\", "/").split("/")[
#                                                  -2] + "TestandWatermarkAcc.png")))
#
#     df = pd.DataFrame(results, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
#     fig, ax = plt.subplots(figsize=(8, 6))
#
#     for name, group in df.groupby("Method Name"):
#         group.plot(1, 2, ax=ax, label="Test acc", linestyle='--', marker='o', color='tab:purple')
#
#     df_adv = pd.DataFrame(results_adv, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
#     ax.set_xlabel("Stealing Dataset Size")
#     ax.set_ylabel("Stolen Model Test and Adversarial Accuracy")
#     for name, group in df_adv.groupby("Method Name"):
#         group.plot(1, 2, ax=ax, label="Watermark acc", linestyle='--', marker='o', color='tab:orange')
#     plt.savefig(image_save_name)
#     file1.close()
#
#     return acc, acc_adv
#
#
#
# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# tf.compat.v1.random.set_random_seed(seed)
#
# finetuned_model_path = "../models/finetuned_finetuning_02-09-2023/true/cifar10_25_25_bestfgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5"
# adv_file_path = "../data/fgsm/cifar10/true/fgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"
# now = datetime.now().strftime("%d-%m-%Y")
# RESULTS_PATH = f"../results/attack_finetuned{now}"
# LOSS_Acc_FOLDER = "losses_acc"
# MODEL_PATH = f"../models/attack_finetuned{now}"
# DATA_PATH = "../data"
#
# if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "true")):
#     os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "true"))
#
# if not os.path.exists(os.path.join(MODEL_PATH, "true")):
#     os.makedirs(os.path.join(MODEL_PATH, "true"))
#
# dataset_name = "cifar10"
#
# option = st.selectbox("With how many number of queries to attack the model?", (250, 500, 1000, 5000, 10000, 20000))
#
# if st.button("Performing Attack on Defended Model"):
#     with st.spinner('In progress...'):
#         acc, acc_adv = model_extraction_attack("cifar10", adv_file_path,
#                                         "cifar10_base_2",
#                                         number_of_queries=[option],
#                                         num_epochs_to_steal= 50, dropout=0,
#                                         optimizer="adam",
#                                         lr=0.01, weight_decay=0,
#                                         model_to_attack_path=finetuned_model_path)
#     st.success("Done!")
#
#     results = ({"Stealing Dataset size": [option], "Test Accuracy": [acc], "Watermark Accuracy": [acc_adv]})
#     # df = pd.DataFrame(columns=['Stealing Dataset Size', 'Test Accuracy', "Watermark Acuuracy"])
#     # df.concat([option, acc, acc_adv])
#     df = pd.DataFrame(results)
#
#     st.table(df)
