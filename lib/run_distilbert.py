import tensorflow as tf
from transformers import (
    TFDistilBertModel,
    DistilBertTokenizer,
    TFDistilBertMainLayer,
    TFDistilBertPreTrainedModel,
)
from transformers.modeling_tf_utils import get_initializer
from adamw_optimizer import CustomSchedule, AdamW
import tensorflow_addons as tfa
from generate_predictions import get_prediction_json
from absl import app
from absl import flags
import time

num_train_examples = 494670


# -----------------------------------------------------------------------------------------------------------------------
# FLAGS
# -----------------------------------------------------------------------------------------------------------------------


FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "training_mode", False, "Do you want to train the model or generate predictions?"
)

flags.DEFINE_bool(
    "use_chkpt", True, "Do you want to use a checkpoint or train from scratch?"
)

flags.DEFINE_string("train_file", None, "File to generate train-dataset from")

flags.DEFINE_string("val_file", None, "File to generate validation(dev)-dataset from")

flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Path to where checkpoints are stored and have to be stored",
)

flags.DEFINE_string(
    "json_output_path", None, "Path to where json predictions are to be stored"
)

flags.DEFINE_string(
    "pred_file",
    None,
    "json file where validations are stored, with-or-without annotations",
)

flags.DEFINE_integer("epochs", 2, "number of epochs")

flags.DEFINE_integer("batch_size", 2, "batch size")

flags.DEFINE_float("init_learning_rate", 3e-5, "initial learning rate")

flags.DEFINE_float(
    "init_weight_decay_rate", 0.01, "init weight decay rate for optimizer"
)

flags.DEFINE_integer("shuffle_buffer_size", 100000, "shuffle buffer size")

flags.DEFINE_integer("best_indexes", 10, "number of best indexes to consider")

answer_types = 5

# -----------------------------------------------------------------------------------------------------------------------
# Read Data
# -----------------------------------------------------------------------------------------------------------------------


def decode_record(record, x):
    """Decodes a record to a TensorFlow example."""

    # parsing one record at a time to a Tensorflow example
    example = tf.io.parse_single_example(record, x)
    for name in list(example.keys()):
        # type conversion for compatibilty
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t
    return example


def read_train_record(tf_record_file, shuffle_buffer_size, batch_size=1):
    """
    Reads tf records into a MapDataset for training

    Parameters: tf_record file and hyperparameters

    Returns: Training dataset
    """

    def x_map(record):
        return (
            {
                "unique_ids": record["unique_ids"],
                "input_ids": record["input_ids"],
                "input_mask": record["input_mask"],
                "segment_ids": record["segment_ids"],
            },
            {
                "start_positions": record["start_positions"],
                "end_positions": record["end_positions"],
                "answer_types": record["answer_types"],
            },
        )

    x = {
        "unique_ids": tf.io.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([512], tf.int64),
        "input_mask": tf.io.FixedLenFeature([512], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([512], tf.int64),
        "start_positions": tf.io.FixedLenFeature([], tf.int64),
        "end_positions": tf.io.FixedLenFeature([], tf.int64),
        "answer_types": tf.io.FixedLenFeature([], tf.int64),
    }

    #  read dataset from record into examples
    dataset = tf.data.TFRecordDataset(tf_record_file).map(
        lambda record: decode_record(record, x)
    )
    # shuffle
    dataset = (
        dataset.shuffle(shuffle_buffer_size) if shuffle_buffer_size != 0 else dataset
    )
    # create batches
    dataset = dataset.batch(batch_size) if batch_size != 0 else dataset
    #  map dataset to features dictionary for ease of access
    dataset = dataset.map(x_map)

    return dataset


def read_val_record(tf_record_file, shuffle_buffer_size, batch_size=1):
    """
    Reads tf records into a MapDataset for validation

    Parameters: tf_record file and hyperparameters

    Returns: Validation dataset
    """

    def x_map(record):
        return {
            "unique_ids": record["unique_ids"],
            "input_ids": record["input_ids"],
            "input_mask": record["input_mask"],
            "segment_ids": record["segment_ids"],
            "token_map": record["token_map"],
        }

    x = {
        "unique_ids": tf.io.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([512], tf.int64),
        "input_mask": tf.io.FixedLenFeature([512], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([512], tf.int64),
        "token_map": tf.io.FixedLenFeature([512], tf.int64),
    }

    #  read dataset from record into examples
    dataset = tf.data.TFRecordDataset(tf_record_file).map(
        lambda record: decode_record(record, x)
    )
    #  shuffle
    dataset = (
        dataset.shuffle(shuffle_buffer_size) if shuffle_buffer_size != 0 else dataset
    )
    # create batches
    dataset = dataset.batch(batch_size) if batch_size != 0 else dataset
    #  map dataset to features dictionary for ease of access
    dataset = dataset.map(x_map)

    return dataset


# -----------------------------------------------------------------------------------------------------------------------
# Define Model
# -----------------------------------------------------------------------------------------------------------------------


class TFNQModel(TFDistilBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        """Initializes model"""
        # initialize pretrained model
        TFDistilBertPreTrainedModel.__init__(self, config, *inputs, **kwargs)

        # set backend as DistilBert
        self.backend = TFDistilBertMainLayer(config, name="distilbert")

        # initialize dropout layers
        self.seq_output_dropout = tf.keras.layers.Dropout(
            kwargs.get("seq_output_dropout_prob", 0.05)
        )
        self.pooled_output_dropout = tf.keras.layers.Dropout(
            kwargs.get("pooled_output_dropout_prob", 0.05)
        )

        # set up classifiers on BERT outputs to give us start and end pos tags, as well as an answer type tag
        self.pos_classifier = tf.keras.layers.Dense(
            2,
            kernel_initializer=get_initializer(config.initializer_range),
            name="pos_classifier",
        )

        self.answer_type_classifier = tf.keras.layers.Dense(
            answer_types,
            kernel_initializer=get_initializer(config.initializer_range),
            name="answer_type_classifier",
        )

    def call(self, inputs, **kwargs):
        """
        Invoked when model called to return logits

        Returns: logits for start token, end token and answer type

        """
        inputs = inputs[:2] if isinstance(inputs, tuple) else inputs
        outputs = self.backend(inputs, **kwargs)

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        # dropout for both outputs
        sequence_output = self.seq_output_dropout(
            sequence_output, training=kwargs.get("training", False)
        )
        pooled_output = self.pooled_output_dropout(
            pooled_output, training=kwargs.get("training", False)
        )
        # splitting into start and end after passing throught classifier built on top of bert
        pos_logits = self.pos_classifier(sequence_output)
        start_pos_logits = pos_logits[:, :, 0]
        end_pos_logits = pos_logits[:, :, 1]

        answer_type_logits = self.answer_type_classifier(pooled_output)

        outputs = (start_pos_logits, end_pos_logits, answer_type_logits)

        return outputs


# -----------------------------------------------------------------------------------------------------------------------
# Initialize Metrics and Optimizer
# -----------------------------------------------------------------------------------------------------------------------


def initialize_acc():
    """Initialize accuracy metrics using Sparse TopK categorical accuracy"""
    start_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    end_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    ans_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    total_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    return total_acc, start_acc, end_acc, ans_acc


def create_optimizer(distilBert):
    """Initializes optimizer"""
    num_train_steps = int(FLAGS.epochs * num_train_examples / FLAGS.batch_size)

    # learning rate schedulers
    schedule = CustomSchedule(
        initial_learning_rate=FLAGS.init_learning_rate,
        decay_steps=num_train_steps,
        end_learning_rate=FLAGS.init_learning_rate,
        power=1.0,
        cycle=True,
        num_warmup_steps=0,
    )
    # schedule = PolynomialDecay(
    #     FLAGS.init_learning_rate,
    #     num_train_steps,
    #     end_learning_rate=0.0001,
    #     power=1.0)

    decay_var_list = []

    for i in range(len(distilBert.trainable_variables)):
        name = distilBert.trainable_variables[i].name
        if any(x in name for x in ["LayerNorm", "layer_norm", "bias"]):
            decay_var_list.append(name)

    # AdamW optimizer, similar to what Google uses
    # return AdamW(
    #     weight_decay=FLAGS.init_weight_decay_rate,
    #     learning_rate=schedule,
    #     beta_1=0.9,
    #     beta_2=0.999,
    #     epsilon=1e-6,
    #     decay_var_list=decay_var_list,
    # )

    # LAMB optimizer, known for training BERT super fast (find it at https://arxiv.org/abs/1904.00962 )
    return tfa.optimizers.LAMB(
        learning_rate=schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-06,
        weight_decay_rate=FLAGS.init_weight_decay_rate,
        name="LAMB",
    )


# -----------------------------------------------------------------------------------------------------------------------
# Find loss and gradients
# -----------------------------------------------------------------------------------------------------------------------


def compute_loss(positions, logits):
    """Finds loss between logits and labels"""

    # implemented the way google defines loss in their bert-joint-baseline paper
    one_hot_positions = tf.one_hot(positions, depth=512, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probs, axis=-1))

    # using sparse categorical cross entropy
    # loss_sparse_cat = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = tf.math.reduce_sum(loss_sparse_cat(positions, logits))
    return loss


def compute_label_loss(labels, logits):
    """Find loss for answer type labels"""
    # the way google defines loss in their bert-joint-baseline paper
    one_hot_labels = tf.one_hot(labels, depth=answer_types, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_mean(tf.reduce_sum(one_hot_labels * log_probs, axis=-1))

    # using sparse categorical cross entropy
    # loss_sparse_cat = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = tf.math.reduce_sum(loss_sparse_cat(labels, logits))
    return loss


def compute_gradient(
    distilBert,
    input_ids,
    input_masks,
    segment_ids,
    start_pos_labels,
    end_pos_labels,
    answer_type_labels,
    train_acc,
    train_acc_start_pos,
    train_acc_end_pos,
    train_acc_ans_type,
):
    """
    Computes gradient based on averaged loss from start token, end token and answer type

    Inputs: features (x), labels (y), accuracy metrics

    Returns: Gradients, accuracy
    """

    with tf.GradientTape() as tape:
        # find loss for all three outputs and average it to find total loss
        (start_pos_logits, end_pos_logits, answer_type_logits) = distilBert(
            (input_ids, input_masks, segment_ids), training=True
        )
        loss_start_pos = compute_loss(start_pos_labels, start_pos_logits)
        loss_end_pos = compute_loss(end_pos_labels, end_pos_logits)
        loss_ans_type = compute_label_loss(answer_type_labels, answer_type_logits)
        total_loss = (loss_start_pos + loss_end_pos + loss_ans_type) / 3.0

    # compute gradient
    gradients = tape.gradient(total_loss, distilBert.trainable_variables)

    #  Update accuracy metrics
    train_acc.update_state(start_pos_labels, start_pos_logits)
    train_acc.update_state(end_pos_labels, end_pos_logits)
    train_acc.update_state(answer_type_labels, answer_type_logits)
    train_acc_start_pos.update_state(start_pos_labels, start_pos_logits)
    train_acc_end_pos.update_state(end_pos_labels, end_pos_logits)
    train_acc_ans_type.update_state(answer_type_labels, answer_type_logits)

    acc = (train_acc, train_acc_start_pos, train_acc_end_pos, train_acc_ans_type)

    return gradients, acc


def checkpt(distilBert, checkpoint_path):
    """Reads checkpoint if present and returns checkpoint manager to store checkpoints if required"""
    ckpt = tf.train.Checkpoint(model=distilBert)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
    # restore latest checkpoint if present
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored")
    else:
        print("No checkpoint found")
    return ckpt_manager


# -----------------------------------------------------------------------------------------------------------------------
# Train model
# -----------------------------------------------------------------------------------------------------------------------


def train(
    distilBert,
    optimizer,
    train_dataset,
    ckpt_manager,
    train_acc,
    train_acc_start_pos,
    train_acc_end_pos,
    train_acc_ans_type,
):
    """
    Trains Model as per configurations

    Parameters: model, dataset, checkpoint manager, and metrics

    Returns: nothing but stores checkpoints
    """
    for epoch in range(FLAGS.epochs):
        # reset metrics at every epoch
        train_acc.reset_states()
        train_acc_start_pos.reset_states()
        train_acc_end_pos.reset_states()
        train_acc_ans_type.reset_states()

        for (instance, (x, y)) in enumerate(train_dataset):
            if instance == num_train_examples:
                break
            # generate x and y
            input_ids, input_masks, segment_ids = (
                x["input_ids"],
                x["input_mask"],
                x["segment_ids"],
            )
            start_pos_labels, end_pos_labels, answer_type_labels = (
                y["start_positions"],
                y["end_positions"],
                y["answer_types"],
            )

            # generate gradients and accuracy
            gradients, acc = compute_gradient(
                distilBert,
                input_ids,
                input_masks,
                segment_ids,
                start_pos_labels,
                end_pos_labels,
                answer_type_labels,
                train_acc,
                train_acc_start_pos,
                train_acc_end_pos,
                train_acc_ans_type,
            )

            # apply gradients
            optimizer.apply_gradients(zip(gradients, distilBert.trainable_variables))

            # print accuracy
            (
                train_acc,
                train_acc_start_pos,
                train_acc_end_pos,
                train_acc_ans_type,
            ) = acc

            if (instance + 1) % 1 == 0:
                print(
                    "Epoch {}, Instances processed {}".format(epoch + 1, instance + 1,)
                )

                print("Overall acc = {:.6f}".format(train_acc.result()))
                print("Start Token acc = {:.6f}".format(train_acc_start_pos.result()))
                print("End Token acc = {:.6f}".format(train_acc_end_pos.result()))
                print("Answer Type acc = {:.6f}".format(train_acc_ans_type.result()))

                print("-" * 100)

        if (epoch + 1) % 1 == 0:
            print(
                "\nSaving checkpoint for epoch {} at {}".format(
                    epoch + 1, ckpt_manager.save()
                )
            )

            print("Overall: loss = acc = {.6f}".format(train_acc.result()))
            print("Start Token: acc = {.6f}".format(train_acc_start_pos.result()))
            print("End Token: acc = {.6f}".format(train_acc_end_pos.result()))
            print("Answer Type: acc = {.6f}".format(train_acc_ans_type.result()))


# -----------------------------------------------------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------------------------------------------------


def main(argv):
    # retrieve datasets
    if FLAGS.training_mode:
        train_dataset = read_train_record(
            FLAGS.train_file, FLAGS.shuffle_buffer_size, FLAGS.batch_size
        )
    else:
        val_dataset = read_val_record(
            FLAGS.val_file, FLAGS.shuffle_buffer_size, FLAGS.batch_size
        )
    print("data retrieved")
    # create model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased-distilled-squad"
    )
    distilBert = TFNQModel.from_pretrained("distilbert-base-uncased-distilled-squad")
    print("Model created")
    if FLAGS.training_mode:
        # get checkpoint if exists
        if FLAGS.use_chkpt:
            # use checkpoint
            ckpt_manager = checkpt(distilBert, FLAGS.checkpoint_path)
        else:
            # create checkpoint manager to store checkpoint
            ckpt = tf.train.Checkpoint(model=distilBert)
            ckpt_manager = tf.train.CheckpointManager(
                ckpt, FLAGS.checkpoint_path, max_to_keep=10
            )
        # define accuracy and loss metrics
        (
            train_acc,
            train_acc_start_pos,
            train_acc_end_pos,
            train_acc_ans_type,
        ) = initialize_acc()
        # create optimizer
        optimizer = create_optimizer(distilBert)
        # train
        print("Training starts....")
        st = time.time()
        train(
            distilBert,
            optimizer,
            train_dataset,
            ckpt_manager,
            train_acc,
            train_acc_start_pos,
            train_acc_end_pos,
            train_acc_ans_type,
        )
        print("Time taken:", time.time() - st)
    else:
        # get checkpoint if exists
        _ = checkpt(distilBert, FLAGS.checkpoint_path)
        print("Getting predictions...")
        # generate predictions.json by converting logits to labels
        get_prediction_json(
            distilBert,
            val_dataset,
            FLAGS.pred_file,
            FLAGS.val_file,
            FLAGS.json_output_path,
            FLAGS.best_indexes,
        )


if __name__ == "__main__":
    app.run(main)
