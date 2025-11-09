"""
@author: Ridhima Garg

Introduction:
    It conatins the simple application based on the streamlit to access the model and data.
"""
import numpy as np
import streamlit as st

import real_model_stealing as ms

st.title('Model Defense from stealing attacks!')

if st.button("Stealing Model"):

    len_steals = [10]
    num_epochs = 10
    env, victim_model, classifier_victim, x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy, x_watermark_numpy, y_watermark_numpy, config = ms.setup_victim_attacker("../configurations/perfect/cifar-to-cifar-ws250-rn34_decay.ini")

    for len_steal in len_steals:
        test_acc, watermark_acc, x_watermark_numpy, watermark_predictions = ms.train_attacker(env, classifier_victim, x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy , x_watermark_numpy, y_watermark_numpy, len_steal, num_epochs, config)

        images = x_watermark_numpy[0:3]

        print(x_watermark_numpy[0].shape)


        for idx, img in enumerate(images):
            st.image(images[idx].reshape(224,224,3), clamp=True, channels= "BGR")
            st.header(np.argmax(watermark_predictions[idx]))



