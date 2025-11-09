import numpy as np
import models_new as md
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import matplotlib as mpl
import seaborn as sns
import matplotlib.font_manager as font_manager

sns.set_context("paper", font_scale=3.0, rc={"lines.linewidth": 2.0})


def pca_dim(fit_data):
    pca = PCA(n_components=2)
    pca.fit(fit_data)

    return pca


def pca_and_plot(x, list_data, results_path, type_model, dataset, time, penultimate_layer, image_save_path, image_legend_save_path):

    pca = pca_dim(x)

    # mpl.rcParams['font.size'] = 12 * 1.3  # Default font size for "paper" is 12pt, scaled by 1.5
    # mpl.rcParams['axes.labelsize'] = 11 * 1.3  # Default axes label size for "paper" is 11pt, scaled by 1.5
    # mpl.rcParams['axes.titlesize'] = 12 * 1.3  # Default axes title size for "paper" is 12pt, scaled by 1.5
    # mpl.rcParams['xtick.labelsize'] = 10 * 1.3  # Default xtick label size for "paper" is 10pt, scaled by 1.5
    # mpl.rcParams['ytick.labelsize'] = 10 * 1.3  # Default ytick label size for "paper" is 10pt, scaled by 1.5
    # mpl.rcParams['legend.fontsize'] = 10 * 1.3  # Default legend font size for "paper" is 10pt, scaled by 1.5
    # mpl.rcParams['lines.linewidth'] = 1.3  # Set the default line width to 1.5



    transformed_data_train = pca.transform(list_data[0])
    transformed_data_target = pca.transform(list_data[2])
    transformed_data_watermark = pca.transform(list_data[3])

    y_train_unique = list(np.unique(list_data[1]))

    x_acc_y = {}

    for e in y_train_unique:
        x_acc_y[e] = transformed_data_train[list_data[1]==e]


    # Plotting PCA output in the middle of the training.
    # color_list = np.concatenate([y , np.repeat(watermark_target, batch_size), np.repeat(10, batch_size)], 0)
    # marker_list = ["."]*y.shape[0] + [","]*batch_size + ["<"]*batch_size
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","darkorange","darkgreen", "darkslategray", "deepskyblue", "purple", "cyan", "blue", "crimson", "bisque", "lawngreen"])
    
    fig = plt.figure()

    # fig, ax = plt.subplots()

    # plot = plt.scatter(transformed_data_train[:,0], transformed_data_train[:,1],  c="purple", marker=".", label="train data")

    c_list = ["darkorange", "darkgreen", "darkslategray", "deepskyblue", "yellow", "cyan", "blue", "crimson", "bisque", "lawngreen"]
    i =0

    for key, value in x_acc_y.items():
        plot = sns.scatterplot(x=value[:320,0], y=value[:320,1],  c=c_list[i], marker=".", label="train data - "+str(key), s=80, edgecolor="none")
        i +=1


    plot = sns.scatterplot(x=transformed_data_target[:,0], y=transformed_data_target[:,1],  c="black", marker=",", label="target", s=50, edgecolor="none")
    plot = sns.scatterplot(x=transformed_data_watermark[:,0], y=transformed_data_watermark[:,1],  c="darkorange", marker="<", label="watermark", s=50, edgecolor="none")

    leg = plt.legend()

    plot.get_legend().set_visible(False)

    plt.savefig(image_save_path, bbox_inches='tight')

    handles, labels = plot.get_legend_handles_labels()

    fig_leg = plt.figure(figsize=(3,2))
    ax_leg = fig_leg.add_subplot(111)
    # Add the legend to this new figure
    ax_leg.legend(handles, labels, loc='center', prop=font_manager.FontProperties(size=10))
    # Hide axes in legend figure
    ax_leg.axis('off')

    # fig1 = fig

    # plt.legend()

    # fig2 = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.legend(*ax.get_legend_handles_labels(), loc='center')
    # ax.axis('off')

    

    # def export_legend(legend, filename="legend.svg", expand=[-5,-5,5,5]):
    #     fig  = legend.figure
    #     fig.canvas.draw()
    #     bbox  = legend.get_window_extent()
    #     bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    #     bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    #     return fig
    #     fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    # legend = export_legend(legend)

    # plot = plt.scatter(transformed_data[:,0], transformed_data[:,1], c= color_list, cmap=cmap, marker=marker_list)
    # plt.legend(handles=plot.legend_elements()[0], labels=list(np.unique(color_list)))
    # plt.show()
    plt.savefig(image_legend_save_path, bbox_inches='tight')

    return plot, ax_leg
    

    plt.savefig(os.path.join(results_path, type_model + "_" + dataset + "_" +time+ "_" + str(penultimate_layer)), dpi=fig.dpi)

def validate_watermark(model, trigger_set, label, num_class, customer_model=None):
    labels = np.zeros([trigger_set.shape[0], num_class])
    labels[:, label] = 1

    # if trigger_set.shape[0] < batch_size:
    #     trigger_data = np.concatenate([trigger_set, trigger_set], 0)[:batch_size]
    # else:
    #     trigger_data = trigger_set
    if customer_model:
        intermediate_output = customer_model(trigger_set)
        predictions = model(intermediate_output)
    else:
        predictions = model(trigger_set)
    # if isinstance(predictions, list):
    #     print("Watermark predictions", np.argmax(predictions[-1], axis=1))
    # else:
    #     print("Watermark predictions", np.argmax(predictions, axis=1))
    error = md.error_rate(labels, predictions)
    return 1 - error


def test_model(model, dataset, model_type, num_class, customer_model=None, watermark_target=None):

    if watermark_target is not None:
        
        print("Test trigger")
        victim_watermark_acc_list = []

        for batch, (trigger_batch_train) in enumerate(dataset):
            victim_watermark_acc_list.append(validate_watermark(
            model, trigger_batch_train, watermark_target, num_class, customer_model))

        victim_watermark_acc = np.average(victim_watermark_acc_list)

        print(f"{model_type} || watermark success: {victim_watermark_acc}")

        return victim_watermark_acc

    else:

        victim_error_list = []

        for batch, (x_batch_test, y_batch_test) in enumerate(dataset): 

            if customer_model:
                intermediate_output = customer_model(x_batch_test)
                predictions = model(intermediate_output)
            
            else:
                predictions = model(x_batch_test)
            error = md.error_rate(y_batch_test, predictions)
            victim_error_list.append(error)

        victim_error = np.average(victim_error_list)

        print(f"{model_type} || validation accuracy: {1 - victim_error}")

        return 1-victim_error
