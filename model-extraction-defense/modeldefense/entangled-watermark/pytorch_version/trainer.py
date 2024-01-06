import torch as t


def pairwise_cos_distance(A):
    "cosine distance between same variable with its transpose"

    normalized_A = t.nn.functional.normalize(A, p=2, dim=1)
    return 1 - t.matmul(normalized_A, t.transpose(normalized_A, 0, 1))  ## may be something can go wrong here.


def snnl_func(x, y, temp, metric='cosine'):
    """
    Soft nearest neighbor calculation based upon the similarity metric choosen (cosine, euclidean).
    """

    x = t.nn.ReLU()(x)
    same_label_mask = t.squeeze(t.eq(y, t.unsqueeze(y, 1)))
    if metric == 'euclidean':
        pass
    elif metric == 'cosine':
        dist = pairwise_cos_distance(t.reshape(x, (x.shape[0], -1)))  ## again here. something wong may be
    else:
        raise NotImplementedError()
    exp = t.clip(t.exp(-(dist / temp)) - t.eye(x.shape[0]), 0, 1)
    prob = (exp / (0.00001 + t.unsqueeze(t.sum(exp, 1), 1))) * same_label_mask
    loss = - t.mean(t.log(0.00001 + t.sum(prob, 1)))
    return loss


class Model_trainer:
    """
    Trainer class which has custom functionality for the model.
    Calculating gradients for trigger generation.
    Calculating soft nearest neighbor loss.
    """

    def __init__(self) -> None:
        pass

    def snnl(self, predictions_list, temp_list, w):

        """
        Soft nearest neighbor loss implementation
        """

        x1 = predictions_list[0]
        x2 = predictions_list[1]
        x3 = predictions_list[2]
        w = w
        inv_temp_1 = 100. / temp_list[0]
        inv_temp_2 = 100. / temp_list[1]  # tf.math.divide(100., self.temp[1])
        inv_temp_3 = 100. / temp_list[2]
        loss1 = snnl_func(x1, w, inv_temp_1)
        loss2 = snnl_func(x2, w, inv_temp_2)
        loss3 = snnl_func(x3, w, inv_temp_3)
        res = [loss1, loss2, loss3]
        return res

    def ce_snnl_loss(self, model, actual, x, temp_list, w, factors_list):

        """
        Combined cross entropy and soft nearest neighbor loss for model training.
        """

        temp_list = t.FloatTensor(temp_list)

        predictions_list = model(x)

        cross_entropy = t.nn.CrossEntropyLoss()

        # print("predictions", t.argmax(predictions_list[-1], dim=1))
        # print("actual", actual)
        # print("Accurcay", len((t.argmax(predictions_list[-1], dim=1) == actual).nonzero())/len(actual))
        ce_loss = cross_entropy(predictions_list[-1], actual.long())

        # print("Cross entropy loss", ce_loss)

        return ce_loss

        if not temp_list.requires_grad:
            temp_list.requires_grad = True

        snnl_loss = self.snnl(predictions_list, temp_list, w)
        soft_nearest_neighbor = factors_list[0] * snnl_loss[0] + factors_list[1] * snnl_loss[1] + factors_list[2] * \
                                snnl_loss[2]
        soft_nearest_neighbor = t.gt(t.sum(w), 0) * soft_nearest_neighbor

        grad = t.autograd.grad(snnl_loss, temp_list, create_graph=True, allow_unused=True)
        # print(grad[0][2])
        return ce_loss - soft_nearest_neighbor, grad[0]

    def error_rate(self, model, x, y):

        """
        Error rate to calculate the misclassifications.
        """

        predictions = model(x)[-1]

        if len(y.shape) > 1:
            y = t.argmax(y, 1)

        mistakes = t.ne(y, t.argmax(predictions, 1))
        return t.mean(mistakes.float())

    def snnl_trigger(self, model, temp_list, w, x):

        """
        Calculated the derivative (for snnl w.r.t input) for the trigger generation
        """

        if not x.requires_grad:
            x.requires_grad = True

        predictions_list = model(x)
        snnl_loss = self.snnl(predictions_list, temp_list, w)

        total_loss = snnl_loss[0] + snnl_loss[1] + snnl_loss[2]
        input = x
        total_loss.retain_grad()
        input.retain_grad()

        grad = t.autograd.grad(total_loss, x, create_graph=True, allow_unused=True)

        return grad[0]

    def ce_trigger(self, model, target_class, x):

        """
        Calculated the derivative (for predictions w.r.t input) for the trigger generation
        """

        if not x.requires_grad:
            x.requires_grad = True

        predictions_list = model(x)

        return t.autograd.grad(t.unbind(predictions_list[-1], axis=1)[target_class][0], x)[0]

# def pairwise_euclid_distance(A):
#     # sqr_norm_A = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 0)
#     sqr_norm_A = A^2
#     sqr_norm_B = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 1)
#     inner_prod = tf.matmul(A, A, transpose_b=True)
#     tile_1 = tf.tile(sqr_norm_A, [tf.shape(A)[0], 1])
#     tile_2 = tf.tile(sqr_norm_B, [1, tf.shape(A)[0]])
#     return tile_1 + tile_2 - 2 * inner_prod