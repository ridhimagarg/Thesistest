# import tensorflow_datasets as tfds

# ds = tfds.load(name="mnist", split="train")

# print(tfds.as_numpy(ds))

# for ex in ds_num

# import onnx

# model = onnx.load("model.onnx")

# print(onnx.checker.check_model(model))

# print(onnx.helper.printable_graph(model.graph))

# import models_new as md
# from keras.utils import plot_model
# import tensorflow as tf

# model = md.Plain_2_conv()

# model = tf.keras.models.load_model("models/EWEVictim_mnist_model1")
# plot_model(model, to_file='model.png')


# tf.keras.utils.plot_model(model.build_graph(), to_file="model.png",
#            expand_nested=True, show_shapes=True)




# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Dense
# visible = Input(shape=(10,))
# hidden1 = Dense(10, activation='relu', trainable=False)(visible)
# hidden2 = Dense(20, activation='relu')(hidden1)
# hidden3 = Dense(10, activation='relu')(hidden2)
# output = Dense(1, activation='sigmoid')(hidden3)
# model = Model(inputs=visible, outputs=output)
# # summarize layers
# print(model.summary())
# # plot graph

# print(model.layers)

# model.layers.pop()

# print(model.layers)

# for layer in model.layers[:-2]:
#     layer.trainable = False

# print(model.summary())

# x = Dense(10,activation='relu')(model.layers[-2].output)
# op1 = x
# x = Dense(5,activation='relu')(x)
# op2 = x
# x = Dense(2,activation='relu')(x)
# # outputs = model.layers[-1](x)

# model = Model(inputs=visible, outputs=[op1, op2, x])

# print(model.layers)




    # for idx, layer in enumerate(model.layers):
        
    #     # print(layer.get_weights())
    #     if len(layer.get_weights()) > 0:
    #         print(layer.get_weights()[1].shape)
    #         ewe_model.layers[idx].set_weights(layer.get_weights())
    #         print(layer.name)



# def train_customer_model1(x_train, y_train, x_test, y_test, epochs, dataset, batch_size):

#     params = {"epochs": epochs, "dataset":dataset}

#     input = Input(shape=(28,28,1))
#     conv1 = Conv2D(filters=32, kernel_size=5, activation=None)(input)
#     relu1 = ReLU()(conv1)
#     pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(relu1)
#     drop1 = Dropout(0.5)(pool1)
#     conv2 = Conv2D(filters=64, kernel_size=3, activation=None)(drop1)
#     relu2 = ReLU()(conv2)
#     pool2 = MaxPooling2D(pool_size=(2,2), strides=2)(relu2)
#     drop2 = Dropout(0.5)(pool2)
#     flatten1 = Flatten()(drop2)
#     dense1 = Dense(128, activation=None)(flatten1)
#     drop3 = Dropout(0.5)(dense1)
#     relu3 = ReLU()(drop3)
#     output = Dense(10, activation=None)(relu3)

#     model = Model(inputs=input, outputs=output)

#     experiment_name = dataset+"Plain_2_conv"

#     # with mlflow.start_run(run_name=experiment_name):

#     for param, param_val in params.items():
#         mlflow.log_param(param, param_val)


#     height = x_train[0].shape[0]
#     width = x_train[0].shape[1]
#     try:
#         channels = x_train[0].shape[2]
#     except:
#         channels = 1
#     num_class = len(np.unique(y_train))

#     y_train = tf.keras.utils.to_categorical(y_train, num_class)
#     y_test = tf.keras.utils.to_categorical(y_test, num_class)

#     num_batch = x_train.shape[0] // batch_size 

#     train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#     train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

#     test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#     test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)


#     loss_epoch = []
#     for epoch in range(1):

#         loss_batch = []
#         for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

#             with tf.GradientTape() as tape:

#                 prediction = model(x_batch_train)

#                 loss_value = md.ce_loss(prediction, y_batch_train)

#             grads = tape.gradient(loss_value, model.trainable_weights)
            
#             optimizer.apply_gradients(zip(grads, model.trainable_weights))

#             loss_batch.append(loss_value)

#         print(f"Loss at {epoch} is {np.mean(loss_batch)}")

#         loss_epoch.append(np.mean(loss_batch))

    
    # plt.figure(figsize=(5,5))
    # plt.plot(list(range(epochs)), loss_epoch, label="Train data acc original training", linestyle='--', marker='o', color='tab:orange')
    # plt.xlabel("epochs")
    # plt.ylabel("CE loss")
    # plt.legend()
    # plt.savefig("Trainoriginalloss.png")

    # mlflow.log_artifact("Trainoriginalloss.png", "Celossoriginal")

    # model.save("models/finetuning"+str("OriginalPlain2Conv"))

    # print("Model input", model.input)

    # intermediate_layer_model = Model(inputs=model.input,
    #                              outputs=model.layers[-7].output)

    # input_shape =  model.layers[-7].output.shape

    # print(input_shape)
    # new_input = Input(shape= (5,5,64))
    # conv1 = Conv2D(filters=64, kernel_size=3, activation=None)(new_input)
    # relu1 = ReLU()(conv1)
    # print(relu1.shape)
    # # pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(relu1)
    # drop1 = Dropout(0.5)(relu1)
    # # conv2 = Conv2D(filters=64, kernel_size=3, activation=None)(drop1)
    # # relu2 = ReLU()(conv2)
    # # pool2 = MaxPooling2D(pool_size=(2,2), strides=2)(relu2)
    # # drop2 = Dropout(0.5)(pool2)
    # print(drop1.shape)
    # flatten1 = Flatten()(drop1)

    # print(flatten1.shape)
    # dense1 = Dense(128, activation=None)(flatten1)
    # drop3 = Dropout(0.5)(dense1)
    # relu3 = ReLU()(drop3)
    # output = Dense(10, activation=None)(relu3)

    # ewe_model = Model(inputs=new_input, outputs=output)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

    # loss_epoch = []
    # for epoch in range(epochs):

    #     loss_batch = []
    #     for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    #         with tf.GradientTape() as tape:

    #             intermediate_output = intermediate_layer_model(x_batch_train)

    #             prediction = ewe_model(intermediate_output)

    #             loss_value = md.ce_loss(prediction, y_batch_train)

    #         grads = tape.gradient(loss_value, ewe_model.trainable_weights)
            
    #         optimizer.apply_gradients(zip(grads, ewe_model.trainable_weights))

    #         loss_batch.append(loss_value)

    #     print(f"Loss at {epoch} is {np.mean(loss_batch)}")

    #     loss_epoch.append(np.mean(loss_batch))

import logging


def test():
    logging.basicConfig(filename="test.log", filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # logging.info("testing")

test()

