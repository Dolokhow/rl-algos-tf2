from rltf2.core.rl_layers import MLPBody
import tensorflow as tf
import numpy as np


class MLPCategorical(tf.keras.Model):

    def __init__(self, num_classes, layers_units, activation='relu', name='mlp_categorical'):
        super(MLPCategorical, self).__init__(name=name)
        self.body = MLPBody(layers_units=layers_units, activation=activation)
        self.output_layer = tf.keras.layers.Dense(num_classes)
        self.softm = tf.keras.layers.Softmax()

    def call(self, inputs, **kwargs):
        raw_features = self.body(inputs=inputs)
        logits = self.output_layer(raw_features)
        probs = self.softm(logits)
        return probs

    def get_config(self):
        pass


def test_categorical_mlp():
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    mlp = MLPCategorical(num_classes=10, layers_units=(256, 128), name='test_mlp')
    # FORWARD PASS TEST
    for epoch in range(100):
        print("Start of epoch %d" % (epoch,))

        for step, batch in enumerate(train_dataset):
            data = batch[0]
            labels = batch[1]
            with tf.GradientTape() as tape:
                probs = mlp(data)
                # Compute  loss
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
                grads = tape.gradient(loss, mlp.trainable_weights)
                optimizer.apply_gradients(zip(grads, mlp.trainable_weights))
            acc_metric(labels, probs)

            if step % 100 == 0:
                print("step %d: accuracy = %.2f" % (step, acc_metric.result()))


if __name__ == '__main__':
    test_categorical_mlp()
    # ones = np.ones((3, 4))
    # shape = ones.shape
    # dummy_state = tf.constant(np.zeros(shape=(1,) + shape, dtype=np.float32))
    # dummy_action = tf.constant(np.zeros(shape=[1, 3], dtype=np.float32))
    # features = tf.concat((dummy_state, dummy_action), axis=1)
    # print()
