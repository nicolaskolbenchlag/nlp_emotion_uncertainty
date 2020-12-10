import numpy as np
import sklearn
import sys

import tensorflow as tf
from sklearn.metrics import precision_score, classification_report, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# TODO: create dynamic outputs/loss function depending on data_pipeline
def y_single_output_wrapper(y):
    # just take the first one
    #return tf.squeeze(tf.one_hot(y[list(y.keys())[0]], depth=2, axis = 1))
    # print(y)
    return y[list(y.keys())[0]]

def quantile_prediction_median_wrapper(predictions, quantiles):
    # receive one list of y_hat for each output, just take the output that predicts the median
    return predictions[quantiles.index(0.5)]

    # We use only the mean for 

def loss_function_wrapper(pred, true, true_uncertain, uncertainty_aware, task_type, y_dim, y_quantilies, hallucinated_vector=None, true_original=None, Han=False, true_Han=None, split_softloss_hardloss=None,
                            Han_compare_to_mean=True,
                            mean_std_split=[1., 1.],
                            Kendall_regression_style=False):




    loss = tf.constant(0.0)
    # if hallucinated_vector is not None:
    #     hallucination_loss = 0.
    #     # Compute hallucination loss
    #     # And also set a weight for the hallucination loss (in the initial configuration)
    #     # experiment with different weights.
    #     loss += hallucination_loss

    
    # loss = tf.Print(loss, [true], "\ntrue is\n")
    # if true_uncertain is not None:
    #     loss = tf.Print(loss, [true_uncertain], "\n true_uncertain is\n")
    # TODO: Could help with some weighted loss? Perhaps on emphasis on the binary task?
    # TODO: dynamic, get labels form data_pipeline
    if Han:
        loss, mean_loss, uncertainty_loss = han_loss(pred, true_Han, true_accepted=true, split_softloss_hardloss=split_softloss_hardloss,
                        compare_to_mean=Han_compare_to_mean,
                        w_mean=mean_std_split[0],
                        w_std=mean_std_split[1])
        return loss, mean_loss, uncertainty_loss
    if Kendall_regression_style:
        loss = kendall_loss_regression_style(pred, true, true_uncertain, true_original, split_softloss_hardloss)
        return loss
    if task_type == 'classification':
        if uncertainty_aware:
            # true = tf.Print(true, [], "\n")
            # true = tf.Print(true, [], "-----------------------")


            # true = tf.Print(true, [true], "\n true is \n")
            # true_uncertain = tf.Print(true_uncertain, [true_uncertain], "\n true uncertain is \n")


            # true = tf.Print(true, [], "\n-----------------------.")
            if y_dim == 2:
                if pred.shape[1] == 4:                        
                    bcloss = kendall_loss(pred, true, true_uncertain, true_original, split_softloss_hardloss)
                elif true_original is None:
                    bcloss = binary_classification_loss_soft(pred,  # data.accept_task.logits
                                                                true,
                                                                true_uncertain,
                                                                split_softloss_hardloss=split_softloss_hardloss)
                else:
                    bcloss= binary_classification_loss_soft_original_labels(pred, true, list(true_original.values()), split_softloss_hardloss)
                if loss.dtype != bcloss.dtype:
                    loss = tf.cast(loss, tf.float32)
                    bcloss = tf.cast(bcloss, tf.float32)
                loss = loss + bcloss
            elif y_dim > 2:
                raise NotImplementedError
        else:
            if y_dim == 2:
                bcloss = binary_classification_loss(pred,  # data.accept_task.logits
                                                         true)
                if loss.dtype != bcloss.dtype:
                    loss = tf.cast(loss, tf.float32)
                    bcloss = tf.cast(bcloss, tf.float32)
                loss = loss + bcloss                                        
            elif y_dim > 2:
                mcloss = mutli_classification_loss(pred,  # data.accept_task.logits
                                                        true)
                if loss.dtype != mcloss.dtype:
                    loss = tf.cast(loss, tf.float32)
                    mcloss = tf.cast(mcloss, tf.float32)
                loss = loss + mcloss
    elif task_type == 'regression':
        if uncertainty_aware:
            # preprocessing
            #
            # TODO: Could help with an auxiliary loss that makes sure low < median < high?
            # TODO: Could help with aux. loss that is doing text-based clustering?
            loss = loss + regression_loss_uncertain(pred, true, true_uncertain)
        else:
            # preprocessing
            #
            # TODO: Could help with an auxiliary loss that makes sure low < median < high?
            # TODO: Could help with aux. loss that is doing text-based clustering?

            loss = loss + regression_loss(pred, true)

    elif task_type == 'quantile_regression':
        if uncertainty_aware:
            loss = loss + quantile_regression_loss(pred, true, true_uncertain, y_quantilies)
        else:
            loss = loss + quantile_regression_loss(pred, true, y_quantilies)

    return loss


def initialize_classification_measures():
    measures = dict()
    measures["train_true"] = None  # classes
    measures["val_true"] = None  # classes
    measures["test_true"] = None  # classes
    measures["test_pred"] = None  # logits
    measures["best_val_pred"] = None  # logits
    measures["pred_train_all_epochs"] = list()
    measures["pred_val_all_epochs"] = list()
    measures["pred_val_soft_all_epochs"] = list()
    measures["test_measure"] = None  # Macro-F1?
    measures["best_val_measure"] = 0.0  # Macro-F1?
    measures["test_measure_dict"] = None
    measures["best_val_measure_dict"] = None


    return measures


def initialize_regression_measures():
    measures = dict()
    measures["train_true"] = None  # value
    measures["val_true"] = None  # value
    measures["test_true"] = None  # value
    measures["test_pred"] = None  # value
    measures["best_val_pred"] = None  # value
    measures["pred_train_all_epochs"] = list()
    measures["pred_val_all_epochs"] = list()
    measures["test_measure"] = None  # RMSE
    measures["best_val_measure"] = np.inf  # RMSE
    measures["test_measure_dict"] = None
    measures["best_val_measure_dict"] = None

    return measures


def binary_classification_loss(pred, true):
    if true.dtype != pred.dtype:
        true = tf.cast(true, tf.float32)
        pred = tf.cast(pred, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true,
                                                   logits=pred)
    return loss


def binary_classification_loss_soft(pred, true, true_soft, split_softloss_hardloss=[0.4, 0.6]):
    # This is the loss function to use if we are using soft labels
    # This does a weighted average of the loss using the hard labels and loss using the soft labels
    # Can experiment with this
    if pred.dtype != true.dtype:
        # Make sure that true and pred can be compared, by ensuring they are of the same datatype
        true = tf.cast(true, tf.float32)
        pred = tf.cast(pred, tf.float32)
    #true_soft = tf.Print(true_soft, [true, true_soft], "\n true and true soft\n")
    if pred.dtype != true_soft.dtype:
        # Make sure that true and pred can be compared, by ensuring they are of the same datatype
        true_soft = tf.cast(true_soft, tf.float32)
        pred = tf.cast(pred, tf.float32)
    loss = (split_softloss_hardloss[1] * tf.nn.sigmoid_cross_entropy_with_logits(labels=true,
                                                          logits=pred) +
            split_softloss_hardloss[0] * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_soft,
                                                          logits=pred))
    return loss

def sigmoid_cross_entropy_with_sorting_out_zerozero(labels, logits):
    loss = 0.
    batch_size = 50
    for i in range(int(batch_size)):
        label = labels[i, :]
        logit = logits[i, :]
        zero = tf.constant([0., 0.], dtype = tf.float32)
        # The entries of "labels" will be equal to [0., 0.] if the given reviewer did not review the paper in question
        # In this case, his review should be sorted out and not considered
        # This is what the following two lines (hopefully) do.
        prima_facie_new_loss = tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit),  tf.float32)
        new_loss = tf.cond(tf.reduce_all(tf.equal(zero, label)), lambda: tf.cast(0., tf.float32), lambda: prima_facie_new_loss, name = "CheckIfReviewerDidReview")
        loss += new_loss
        #if i ==1:
        #    loss = tf.Print(loss, [label, logit, new_loss], "Label, logit, newloss is ")

        # reviewer_didnt_hand_in_a_rating = tf.reduce_all(tf.math.equal(label[0], zero)) and tf.reduce_all(tf.math.equal(label[1], zero))
        # if reviewer_didnt_hand_in_a_rating:
        #     print(f"\n sorting out {label} \n")
        #     continue
        # else:
        #     loss += tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
        #     print(f"Loss shape is {tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit).shape}")
    return loss

def binary_classification_loss_soft_original_labels(pred, true_acceptance, true_original_labels, split_softloss_hardloss=[0.4, 0.6]):
    """
    This function takes predictions, their corresponding acceptance status, and a list containing
    What each reviewer recommended in the case of each of the datapoints and computes loss
    as a weighted average of the cross entropy of the prediction with the true acceptance
    and the cross entropy of the prediction with each of the reviewers recommendations
    """
    loss = 0.
    for original_label in true_original_labels:
        #original_label = tf.Print(original_label, [original_label], "original label")
        # Make sure that truth and prediction can be compared, by ensuring they are of the same datatype
        if pred.dtype != true_acceptance.dtype:
            true_acceptance = tf.cast(true_acceptance, tf.float32)
            pred = tf.cast(pred, tf.float32)
        if pred.dtype != original_label.dtype:            
            original_label = tf.cast(original_label, tf.float32)
            pred = tf.cast(pred, tf.float32)
        # Compute the cross entropy of the reviewer's label and the prediction and add it to the loss
        # loss += tf.nn.sigmoid_cross_entropy_with_logits(labels=original_label,
        #                                                logits=pred)
        loss += sigmoid_cross_entropy_with_sorting_out_zerozero(labels=original_label,
                                                               logits=pred)
        # loss += tf.nn.sigmoid_cross_entropy_with_logits(labels=original_label, logits=pred)
        


        # recording_pred_loss_pairs["original_label"] = true_original_labels[:4]
        # recording_pred_loss_pairs["prediction"] = pred[:4]
        # recording_pred_loss_pairs["loss"] = loss[:4]

    # Compute the loss as a weighted average of comparing the prediction against
    if split_softloss_hardloss is None:
        split_softloss_hardloss = [0.4, 0.6]
    loss = (split_softloss_hardloss[1] * tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_acceptance,
                                                        logits=pred), tf.float32) +
            split_softloss_hardloss[0] * loss)
    return loss

def kendall_loss(pred, true, true_uncertain, true_original, split_softloss_hardloss=None):
    # This should take a tensor of shape 4 as an input,
    # where the first two dimensions are the predicted mean, and the
    # second two dimensions the predicted uncertainty / variance / std
    # And then compute the loss according to the Kendall paper
    # The following code is taken from Piotr Ewiak's individual project

    if true_original is not None:
        original_label_list = list(true_original.values())
    
    if pred.dtype != tf.float32:
        pred = tf.cast(pred, tf.float32)
    T = 50
    mean = pred[:, :2]
    log_var = pred[:, 2:]
    precision = tf.exp(-log_var)
    var = tf.exp(log_var)
    std = tf.sqrt(var)
    if mean.dtype != tf.float32:
        mean = tf.cast(mean, tf.float32)


    loss = tf.constant([0.])
    # loss = tf.Print(loss, [pred], "\n pred is\n")
    # loss = tf.Print(loss, [true], "\n binary truth is \n")

    
    # try:
    #     loss = tf.Print(loss, [true_uncertain], "\n uncertain truth is\n")
    # except:
    #     print("true uncertain is none")
    # try:
    #     loss = tf.Print(loss, original_label_list, "\n true original is ")
    # except:
    #     print("true original is none")



    # Loss code chunk.
    dist = tf.distributions.Normal(loc=mean, scale=std)
    noisy_pred = dist.sample([T])
    true_expanded = tf.expand_dims(true, axis=0)
    # noisy_pred = tf.Print(noisy_pred, [true_expanded[0][0], true_expanded[0][1]], f"\nShape of te is {true_expanded.shape} true_expanded is \n")


    second_multiplicative_component = noisy_pred - tf.log(tf.reduce_sum(tf.exp(noisy_pred), 2, keep_dims=True))
    # true_expanded = tf.Print(true_expanded, [second_multiplicative_component[0][0], second_multiplicative_component[0][1]], f"\n shape of smc is {second_multiplicative_component.shape} the second multiplicative component is \n")
    
    if true_uncertain is not None:
        multiplication_result = tf.multiply(true_uncertain, second_multiplicative_component)
    elif true_original is not None:
        for original_label in original_label_list:
            multiplication_result = tf.multiply(original_label, second_multiplicative_component)
            intermediate_loss = tf.reduce_sum(
                                multiplication_result, 2)
            intermediate_loss = tf.reduce_mean(intermediate_loss, 0)  # Average over samples.
            intermediate_loss = - tf.reduce_sum(intermediate_loss, 0)
            # intermediate_loss = tf.Print(intermediate_loss, [intermediate_loss], "\n int loss is")
            loss += intermediate_loss
            # loss = tf.Print(loss, [loss], "\n loss is \n")
        return loss
    else:
        multiplication_result = tf.multiply(true_expanded, second_multiplicative_component)
    # true_expanded = tf.Print(true_expanded, [multiplication_result[0][0], multiplication_result[0][1]], f"\n mr shape {multiplication_result.shape} multiplication result:")

    # reduced_sum = tf.reduce_sum(multiplication_result, 2)
    # true_expanded = tf.Print(true_expanded, [reduced_sum[0], reduced_sum[1]], f"\nreduced sum shape is {reduced_sum.shape} and reduced sum is\n")


    # The "multiplication_result" is the result of multiplying a tensor like [-1.123123 0.34343]
    # with a tensor like [0 1] or [1 0] - the result would be, in the first case, [0. 0.34343] and in the second case [-1.123123 0.]
    # so, the following line, where we apply tf.reduce_sum, will do nothing else than just produce for each of these 2-tuples
    # the nonzero component
    loss = tf.reduce_sum(
        multiplication_result, 2)
    loss = tf.reduce_mean(loss, 0)  # Average over samples.
    loss = - tf.reduce_sum(loss, 0)  # Sum over minibatch.

    if split_softloss_hardloss is not None:
        loss = (split_softloss_hardloss[1] * tf.nn.sigmoid_cross_entropy_with_logits(labels=true,
                                                            logits=mean) +
                split_softloss_hardloss[0] * loss)

    return loss
def kendall_loss_regression_style(pred, true, true_uncertain, true_original, split_softloss_hardloss=None):
    # This should take a tensor of shape 3 as an input,
    # where the first two dimensions are the predicted mean, and the
    # third dimensions the predicted uncertainty / variance / std
    # And then compute the loss according to the Kendall paper;
    # but not in the way they describe their "uncrtainty loss" for classification,
    # but the way they do it in the case of regression; i.e.: simply using
    # the inverse of the predicted variance as a weight for the loss
    # while simultaneously incentivising it to stay small.
    # The following code is inspired by Piotr Ewiak's individual project

    if true_original is not None:
        original_label_list = list(true_original.values())
    
    if pred.dtype != tf.float32:
        pred = tf.cast(pred, tf.float32)


    mean = pred[:, :2]
    log_var = pred[:, -1]
    precision = tf.exp(-log_var)
    var = tf.exp(log_var)
    if mean.dtype != tf.float32:
        mean = tf.cast(mean, tf.float32)
    

    if true_uncertain is not None:
        first_component = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_uncertain,
                                                                    logits=mean)
        first_component = tf.reduce_sum(first_component, axis=1)
    elif true_original is not None:
        loss = tf.constant([0.])
        for original_label in original_label_list:
            zero = tf.constant([0., 0.], dtype = tf.float32)
            # The entries of "labels" will be equal to [0., 0.] if the given reviewer did not review the paper in question
            # In this case, his review should be sorted out and not considered
            # This is what the following two lines (hopefully) do.
            prima_facie_new_loss = tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(labels=original_label, logits=mean),  tf.float32)
            new_loss = tf.cond(tf.reduce_all(tf.equal(zero, original_label)), lambda: 0. * prima_facie_new_loss, lambda: prima_facie_new_loss, name = "CheckIfReviewerDidReview")
            first_component = tf.reduce_sum(new_loss, axis=1)
            first_component = 0.5 * tf.multiply(precision, first_component)
            first_component = tf.reduce_mean(first_component)

            second_component = 0.5 * tf.reduce_sum(var)

            loss += first_component + second_component
    
        if split_softloss_hardloss is not None:
            loss = (split_softloss_hardloss[1] * tf.nn.sigmoid_cross_entropy_with_logits(labels=true,
                                                                logits=mean) +
                    split_softloss_hardloss[0] * loss)

        return loss

    else:
        first_component = tf.nn.sigmoid_cross_entropy_with_logits(labels=true,
                                                                    logits=mean)
        first_component = tf.reduce_sum(first_component, axis=1)

    first_component = 0.5 * tf.multiply(precision, first_component)
    first_component = tf.reduce_mean(first_component)

    second_component = 0.5 * tf.reduce_sum(var)

    #second_component = tf.Print(second_component, [second_component], "\nsecond comp \n")

    loss = first_component + second_component


    if split_softloss_hardloss is not None:
        loss = (split_softloss_hardloss[1] * tf.nn.sigmoid_cross_entropy_with_logits(labels=true,
                                                            logits=mean) +
                split_softloss_hardloss[0] * loss)

    return loss


def han_loss(pred_softlabel, true_softlabel, true_accepted, split_softloss_hardloss = [0.4, 0.6], w_mean=1., w_std=1.,
            compare_to_mean=True):
    """
    This implements the loss function for soft labels according to Han et al.
    It is the weighted sum of the mean squared errof of the predicted mean
    and the mean squared error of the predicted std
    """
    pred_mean = tf.cast(pred_softlabel[:, :2], tf.float32)
    pred_std = tf.cast(pred_softlabel[:, 2:], tf.float32)
    true_mean = tf.cast(true_softlabel[:, :2], tf.float32)
    true_std = tf.cast(true_softlabel[:, 2:], tf.float32)
    true_accepted = tf.cast(true_accepted, tf.float32)

    if compare_to_mean:
        mean_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_mean,
                                            logits=pred_mean)
    # If we have set compare_to_mean to False,
    # we compute the "mean" component of the loss, i.e. the component
    # that should capture how close the prediction is to the target,
    # by comparing to the editorial decision instead of the mean of the raters
    else:
        mean_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_accepted,
                                            logits=pred_mean)

    mean_loss = w_mean * mean_loss
    std_loss = tf.losses.mean_squared_error(labels=true_std,
                                            predictions=pred_std
                                            )
    
    std_loss = w_std * std_loss

    han_loss= mean_loss + std_loss
    loss = (split_softloss_hardloss[1] * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_accepted,
                                                          logits=pred_mean) +
            split_softloss_hardloss[0] * han_loss)
    return loss, mean_loss, std_loss





def mutli_classification_loss(pred, true):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true,
                                                      logits=pred)
    return loss

def regression_loss(pred, true):
    loss = tf.losses.mean_squared_error(labels=true,
                                        predictions=pred)
    return loss

def quantile_regression_loss(predictions, labels, quantiles):

    losses = []
    # TO-DO: Expend by one dimension?
    for i, quantile in enumerate(quantiles):
       error = tf.subtract(labels, predictions[i])
       loss = tf.reduce_mean(tf.maximum(quantile*error, (quantile-1)*error), axis=-1)
       # clipping: https://arxiv.org/pdf/1808.08798.pdf
       # loss = tf.reduce_mean(quantile*error + tf.clip(-error, tf.epsilon(), np.inf), axis=-1)

       losses.append(loss)

    combined_loss = tf.reduce_mean(tf.add_n(losses))

    return combined_loss


def regression_loss_uncertain(pred, true, true_std):

    loss = tf.losses.mean_squared_error(labels=true,
                                        predictions=pred)

    number_of_mc_samples = 200

    for mc in range(1, number_of_mc_samples):
        true_eff = np.random.random(loc=true, scale=true_std)

        loss += tf.losses.mean_squared_error(labels=true_eff,
                                             predictions=pred)
    loss = loss / number_of_mc_samples

    return loss


def quantile_regression_loss_uncertain(predictions, labels, labels_std, quantiles):

    losses = []
    # TO-DO: Expend by one dimension?
    for i, quantile in enumerate(quantiles):
       error = tf.subtract(labels, predictions[i])
       loss = tf.reduce_mean(tf.maximum(quantile*error, (quantile-1)*error), axis=-1)
       # clipping: https://arxiv.org/pdf/1808.08798.pdf
       # loss = tf.reduce_mean(quantile*error + tf.clip(-error, tf.epsilon(), np.inf), axis=-1)

       losses.append(loss)

    combined_loss = tf.reduce_mean(tf.add_n(losses))

    number_of_mc_samples = 200

    for mc in range(1, number_of_mc_samples):
        labels_eff = np.random.random(loc=labels, scale=labels_std)

        losses = []
        # TO-DO: Expend by one dimension?
        for i, quantile in enumerate(quantiles):
            error = tf.subtract(labels_eff, predictions[i])
            loss = tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error), axis=-1)
            # clipping: https://arxiv.org/pdf/1808.08798.pdf
            # loss = tf.reduce_mean(quantile*error + tf.clip(-error, tf.epsilon(), np.inf), axis=-1)

            losses.append(loss)

        combined_loss += tf.reduce_mean(tf.add_n(losses))

    combined_loss = combined_loss / number_of_mc_samples

    return combined_loss


def display_stats_classification(labels, predictions, verbose=True):
    """
    Takes in the true label and prediction vectors and displays the model accuracy and
    other stats
    """

    precision = precision_score(labels, predictions, average='micro')

    print("Precision: {}".format(precision))

    if verbose:
        classification_stats = classification_report(labels, predictions)

        print(classification_stats)

    return precision

def display_stats_regression(labels, predictions, task_type, verbose=True):
    """
    Takes in the true label and prediction vectors and displays the model accuracy and
    other stats
    """

    if verbose:
        print("MSE: ", mean_squared_error(labels, predictions))
        print("R-squared: ", r2_score(labels, predictions))

    return None


def stable_softmax(X):
    exps = np.exp(X - np.max(X, 1).reshape((X.shape[0], 1)))
    return exps / np.sum(exps, 1).reshape((X.shape[0], 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_precision(labels, predictions, type='micro'):
    """
    Returns the precision based on the list of predictions and true labels given the type of precision
    :param labels: true classes of the samples
    :param predictions: predicted labels outputted by the model
    :param type: type of precision needed {'micro', 'macro', 'weighted'}
    :return: the precision score
    """
    return precision_score(labels, predictions, average=type)


def get_classification_measures(true, pred_logits):
    # Make one hot.
    # true_indicator = np.zeros((true.size, true.max() + 1))  # TODO: This is hacky and may crash
    # true_indicator[np.arange(true.size), true] = 1

    true_indicator = true

    measures = dict()

    # Accuracy.
    pred_prob = stable_softmax(pred_logits)

    sum_prob = np.sum(pred_prob, axis=1)

    # pred_labels = (pred_prob.argmax(1)[:, None] == np.arange(pred_prob.shape[1])).astype(int)
    true_labels = np.argmax(true, axis=-1)
    pred_labels = np.argmax(pred_logits, axis=-1)

    # print(np.hstack([true_labels.reshape((-1, 1)), pred_labels.reshape((-1, 1))]))

    # acc = sklearn.metrics.accuracy_score(true_indicator, pred_labels, normalize=True, sample_weight=None)
    acc = sklearn.metrics.accuracy_score(true_labels, pred_labels, normalize=True, sample_weight=None)
    measures["accuracy"] = acc
    balanced_acc = sklearn.metrics.balanced_accuracy_score(true_labels, pred_labels)
    measures["balanced_accuracy"] = balanced_acc
    precision = sklearn.metrics.precision_score(true_labels, pred_labels, sample_weight=None)
    measures["precision"] = precision
    recall = sklearn.metrics.recall_score(true_labels, pred_labels, sample_weight=None)
    measures["recall"] = recall
    confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, pred_labels, normalize="all")
    measures["confusion_matrix"] = confusion_matrix


    # Macro-F1.
    # macro_f1 = sklearn.metrics.f1_score(true_indicator, pred_labels, average="macro")
    macro_f1 = sklearn.metrics.f1_score(true_labels, pred_labels, average="macro")
    measures["macro-f1"] = macro_f1

    # AU-ROC.
    au_roc_macro = sklearn.metrics.roc_auc_score(true_indicator, pred_prob, average="macro")
    au_roc_micro = sklearn.metrics.roc_auc_score(true_indicator, pred_prob, average="micro")
    measures["au-roc-macro"] = au_roc_macro
    measures["au-roc-micro"] = au_roc_micro

    # AU-PRC
    au_prc_macro = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average="macro")
    au_prc_micro = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average="micro")
    measures["au-prc-macro"] = au_prc_macro
    measures["au-prc-micro"] = au_prc_micro

    print("Macro-F1     ", measures["macro-f1"])
    print("Micro-F1     ", measures["accuracy"])

    print("Macro-AU-ROC ", measures["au-roc-macro"])
    print("Micro-AU-ROC ", measures["au-roc-micro"])
    print("Macro-AU-PRC ", measures["au-prc-macro"])
    print("Micro-AU-PRC ", measures["au-prc-micro"])

    return measures


def get_regression_measures(labels, predictions):
    measures = dict()
    measures["mse"] = mean_squared_error(labels, predictions)
    measures["rmse"] = np.sqrt(mean_squared_error(labels, predictions))
    measures["mae"] = mean_absolute_error(labels, predictions)
    measures["r_squared"] = r2_score(labels, predictions)

    print("MSE  ", measures["mse"])
    print("RMSE ", measures["rmse"])
    print("MAE  ", measures["mae"])
    print("R^2  ", measures["r_squared"])

    return measures


def get_f1(labels, predictions, type='macro'):
    return f1_score(labels, predictions, average=type)


def get_mse(labels, predictions):
    return sklearn.metrics.mean_squared_error(labels, predictions)


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def final_score(reviewer_confidence,
                novelty_originality,
                technical_correctness,
                clarity_of_presentation,
                reproducibility,
                quality_of_references,
                overall_recommendation):

    final_score = 1.0 + ((reviewer_confidence + 26.0) / (4 + 26)) * 5 * (0.1 * (novelty_originality - 1.0) / 3.0 +
                                                                         0.05 * (technical_correctness- 1.0) / 3.0 +
                                                                         0.05 * (clarity_of_presentation- 1.0) / 3.0 +
                                                                         0.05 * (reproducibility- 1.0) / 3.0 +
                                                                         0.1 * (quality_of_references- 1.0) / 3.0 +
                                                                         0.65 * (overall_recommendation- 1.0) / 5.0)

    return final_score


def print_tensor(t):
    sess = tf.compat.v1.Session()
    with sess.as_default():
        print_op = tf.print(t, output_stream=sys.stdout)
    with tf.control_dependencies([print_op]):
        tripled_tensor = t * 3
    sess.run(tripled_tensor)


def plot_PR_curve(true, pred_logits, filepath="pr-plots/plot1"):
    # code from https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc

    from matplotlib import pyplot


    from random import random
    from random import randint

    pyplot.clf()


    testy = true[:, 1]

    lr_probs = pred_logits[:, 1]

    noskill = [len(testy[testy==1]) / len(testy) + 0.00000000000001*random() for x in lr_probs]

    noskill_lr_precision, noskill_lr_recall, noskill_thresholds = precision_recall_curve(testy, noskill)


    lr_precision, lr_recall, thresholds = precision_recall_curve(testy, lr_probs)


    lr_auc = auc(lr_recall, lr_precision)
    noskill_lr_auc = auc(noskill_lr_recall, noskill_lr_precision)

    pyplot.plot(lr_recall, lr_precision, marker='.', label='With learning')
    pyplot.plot(noskill_lr_recall, noskill_lr_precision, marker='.', label = 'Baseline without learning')

    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()


    pyplot.savefig(filepath)


    pyplot.clf()







def print_tensor(t):
    sess = tf.compat.v1.Session()
    with sess.as_default():
        print_op = tf.print(t, output_stream=sys.stdout)
    with tf.control_dependencies([print_op]):
        tripled_tensor = t * 3
    sess.run(tripled_tensor)


def plot_ROC_curve(true, pred_logits, filepath="roc-plots/plot1.png"):
    # code from https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score

    from matplotlib import pyplot


    from random import random
    from random import randint

    pyplot.clf()

    testy = true[:, 1]

    lr_probs = pred_logits[:, 1]

    ns_probs = [len(testy[testy==1]) / len(testy) for x in lr_probs]

    ns_auc, learned_auc= roc_auc_score(testy, ns_probs), roc_auc_score(testy, lr_probs)

    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)

    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label = 'Baseline without training')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label= 'With learning')

    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()

    pyplot.savefig(filepath)
    pyplot.clf()