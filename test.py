""" Script for testing trained FilletRec on test dataset"""

import time
import tensorflow as tf
import numpy as np

from src.helper import dataloader as dataloader
from src.analysis import analysis_report_filletData
from src.network import FilletRecGCN as FilletGCN


def test_step(x, y):
    test_logits = model(x, training=False)
    loss_value = loss_fn(y, test_logits)

    y_true = np.argmax(y.numpy(), axis=1)
    y_pred = np.argmax(test_logits.numpy(), axis=1)

    test_loss_metric.update_state(loss_value)
    test_acc_metric.update_state(y, test_logits)
    test_precision_metric.update_state(y, test_logits)
    test_recall_metric.update_state(y, test_logits)

    return y_true, y_pred


if __name__ == '__main__':
    # User defined parameters.
    filters=[64,64,64]
    num_classes = 2
    num_layers = 3
    units = 64
    out_dim=32
    learning_rate = 1e-2
    dropout_rate = 0.3

    checkpoint_path="checkpoint\edge_layers_3_units_64_epochs_20_dim_32_date_2025-08-26.weights.h5"
    
    test_set_path = "data/test_batch_5000.h5"

    model = FilletGCN(units=units, out_channel=out_dim,filter=filters,rate=dropout_rate, num_classes=num_classes, num_layers=num_layers)
    model.build(input_shape=[
    (None, 50),  # V_1
    (None,1),    # Width attr
    (None,1),    # Angle attr
    (None, None),              # A_1
    (None, None),              # E_1
    (None, None),              # E_2
    (None, None)               # E_3
])
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    test_loss_metric = tf.keras.metrics.Mean()
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    test_precision_metric = tf.keras.metrics.Precision()
    test_recall_metric = tf.keras.metrics.Recall()

    model.load_weights(checkpoint_path)
    test_dataloader = dataloader(test_set_path)

    y_true_total = []
    y_pred_total = []

    start_time = time.time()

    for x_batch_test, y_batch_test in test_dataloader:
        one_hot_y = tf.one_hot(y_batch_test, depth=num_classes)
        y_true, y_pred = test_step(x_batch_test, one_hot_y)

        y_true_total = np.append(y_true_total, y_true)
        y_pred_total = np.append(y_pred_total, y_pred)

    print("Time taken: %.2fs" % (time.time() - start_time))

    analysis_report_filletData(y_true_total, y_pred_total)
    test_loss = test_loss_metric.result()
    test_acc = test_acc_metric.result()
    test_precision = test_precision_metric.result()
    test_recall = test_recall_metric.result()

    test_loss_metric.reset_states()
    test_acc_metric.reset_states()
    test_precision_metric.reset_states()
    test_recall_metric.reset_states()
    
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

    print(f"Test loss={test_loss}, Test acc={test_acc}, Precision={test_precision}, F1={test_f1}, Recall={test_recall}")