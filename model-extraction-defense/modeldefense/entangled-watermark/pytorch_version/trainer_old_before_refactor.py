import torch as t


def pairwise_cos_distance(A):
    # normalized_A = tf.nn.l2_normalize(A, 1)
    normalized_A = t.nn.functional.normalize(A, p=2, dim=1)
    # return 1 - tf.matmul(normalized_A, normalized_A, transpose_b=True)
    return 1 - t.matmul(normalized_A, t.transpose(normalized_A, 0, 1))  ## may be something can go wrong here.


def snnl_func(x, y, temp, metric='cosine'):
    x = t.nn.ReLU()(x)
    # same_label_mask = tf.cast(tf.squeeze(tf.equal(y, tf.expand_dims(y, 1))), tf.float32)
    same_label_mask = t.squeeze(t.eq(y, t.unsqueeze(y, 1)))
    if metric == 'euclidean':
        # dist = pairwise_euclid_distance(tf.reshape(x, [tf.shape(x)[0], -1]))
        pass
    elif metric == 'cosine':
        # dist = pairwise_cos_distance(tf.reshape(x, [tf.shape(x)[0], -1]))
        dist = pairwise_cos_distance(t.reshape(x, (x.shape[0], -1)))  ## again here. something wong may be
    else:
        raise NotImplementedError()
    exp = t.clip(t.exp(-(dist / temp)) - t.eye(x.shape[0]), 0, 1)
    # exp = tf.clip_by_value(tf.exp(-(dist / t)) - tf.eye(tf.shape(x)[0]), 0, 1)
    # prob = (exp / (0.00001 + tf.expand_dims(tf.reduce_sum(exp, 1), 1))) * same_label_mask
    prob = (exp / (0.00001 + t.unsqueeze(t.sum(exp, 1), 1))) * same_label_mask
    # loss = - tf.reduce_mean(tf.math.log(0.00001 + tf.reduce_sum(prob, 1)))
    loss = - t.mean(t.log(0.00001 + t.sum(prob, 1)))
    return loss


class Model_trainer:

    def __init__(self) -> None:
        pass

    def snnl(self, predictions_list, temp_list, w):
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

        predictions_list = model(x)

        cross_entropy = t.nn.CrossEntropyLoss()

        ce_loss = cross_entropy(predictions_list[-1], actual.long())

        if not temp_list.requires_grad:
            temp_list.requires_grad = True
        snnl_loss = self.snnl(predictions_list, temp_list, w)

        soft_nearest_neighbor = factors_list[0] * snnl_loss[0] + factors_list[1] * snnl_loss[1] + factors_list[2] * \
                                snnl_loss[2]

        # soft_nearest_neighbor = tf.cast(tf.greater(tf.math.reduce_mean(self.w), 0), tf.float32) * soft_nearest_neighbor

        soft_nearest_neighbor = t.gt(t.sum(w), 0) * soft_nearest_neighbor

        # snnl_loss.retain_grad()
        # temp_list.retain_grad()

        grad = t.autograd.grad(snnl_loss, temp_list, create_graph=True, allow_unused=True)
        print(grad[0][2])

        return ce_loss - soft_nearest_neighbor, grad[0]

    def error_rate(self, model, x, y):

        predictions = model(x)[-1]

        if len(y.shape) > 1:
            y = t.argmax(y, 1)

        print(y)
        print(t.argmax(predictions, 1))

        mistakes = t.ne(y, t.argmax(predictions, 1))

        return t.mean(mistakes.float())

    def snnl_trigger(self, model, temp_list, w, x):
        if not x.requires_grad:
            x.requires_grad = True

        predictions_list = model(x)

        # print(predictions_list[-1])

        # first = t.tensor(predictions_list[-1][0][0], requires_grad=True)
        # second = t.tensor(x, requires_grad=True)

        # print("here", t.autograd.grad(first, second, create_graph=True, allow_unused=True))

        snnl_loss = self.snnl(predictions_list, temp_list, w)

        total_loss = snnl_loss[0] + snnl_loss[1] + snnl_loss[2]
        input = x
        total_loss.retain_grad()
        input.retain_grad()

        grad = t.autograd.grad(total_loss, x, create_graph=True, allow_unused=True)

        return grad[0]

    def ce_trigger(self, model, target_class, x):

        if not x.requires_grad:
            x.requires_grad = True

        predictions_list = model(x)

        # print("trigger", t.unbind(predictions_list[-1], axis=1))
        # grads = []
        # for val in t.unbind(predictions_list[-1], axis=1)[target_class]:
        #     val.retain_grad()
        #     grads.append(t.autograd.grad(val, x, create_graph=True, allow_unused=True)[0])

        # print(grads[0].shape)
        # print(t.cat(grads, dim=1).shape)
        # return t.cat(grads, dim=1)

        return t.autograd.grad(t.unbind(predictions_list[-1], axis=1)[target_class][0], x)[0]
        # return tf.gradients(tf.unstack(predictions_list[-1], axis=1)[target_class], x)

# def pairwise_euclid_distance(A):
#     # sqr_norm_A = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 0)
#     sqr_norm_A = A^2
#     sqr_norm_B = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 1)
#     inner_prod = tf.matmul(A, A, transpose_b=True)
#     tile_1 = tf.tile(sqr_norm_A, [tf.shape(A)[0], 1])
#     tile_2 = tf.tile(sqr_norm_B, [1, tf.shape(A)[0]])
#     return tile_1 + tile_2 - 2 * inner_prod