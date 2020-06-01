# -----------------------------------------------------------------------------------------------------------------------------------------------
# I have used Google's formats for doc strings and formatting. I have left the commented versions of
# some approaches I tried that didn't work as well to give more of an insight into my process.
# Mostly, I have experimented with the optimizers, batch size, learning rate and training set size.
# I had to be selective in my parameter tuning here due to resource constraints.
# If you can't make sense of something, feel free to reach me at sr2259@cornell.edu
# -----------------------------------------------------------------------------------------------------------------------------------------------


from transformers.modeling_tf_utils import get_initializer
from generate_predictions import get_prediction_json
from adamw_optimizer import CustomSchedule, AdamW
from transformers import create_optimizer as co
import tensorflow_addons as tfa
from transformers import (
    TFDistilBertPreTrainedModel,
    TFDistilBertMainLayer,
    DistilBertTokenizer,
    TFDistilBertModel,
)
import tensorflow as tf
from absl import flags
from absl import app
import time

model_name = "distilbert-base-uncased-distilled-squad"

# -----------------------------------------------------------------------------------------------------------------------
# FLAGS
# -----------------------------------------------------------------------------------------------------------------------

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "training_mode", False, "Do you want to train the model or generate predictions?"
)

flags.DEFINE_bool("clipped", False, "Use entire training set or clip it?")

flags.DEFINE_integer("len_train", 494670, "How many training batches to use?")

flags.DEFINE_bool(
    "use_chkpt", True, "Do you want to use a checkpoint or train from scratch?"
)

flags.DEFINE_string("train_file", None, "TF record file to generate train-dataset from")

flags.DEFINE_string(
    "val_file", None, "TF record file to generate validation(dev)-dataset from"
)

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
    "jsonl.gz file where validations are stored, with-or-without annotations",
)

flags.DEFINE_integer("epochs", 2, "number of epochs")

flags.DEFINE_integer("batch_size", 2, "batch size")

flags.DEFINE_float("init_learning_rate", 3e-5, "initial learning rate")

flags.DEFINE_float(
    "init_weight_decay_rate", 0.01, "initial weight decay rate for optimizer"
)

flags.DEFINE_integer("shuffle_buffer_size", 100000, "shuffle buffer size")

flags.DEFINE_integer("best_indexes", 10, "number of best start/end indexes to consider")

# answer types set to 5 for "Unknown, Long, Short, Yes, No"
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

    Args:
        tf_record_file: tf_record training file
        shuffle_buffer_size: A chunk of this size of the dataset is shuffled for randomization
        batch_size: number of batches of consecutive data to be formed

    Returns: Training dataset as a tuple with x(features) and y(labels)
    """

    def x_map(record):
        return (
            {
                # FEATURES
                # unique_ids for every example
                "unique_ids": record["unique_ids"],
                # input_ids corresponding to tokens in the vocabulary
                "input_ids": record["input_ids"],
                # to make sure every input is same seq length
                "input_mask": record["input_mask"],
                # segment_ids to break input into different inputs
                "segment_ids": record["segment_ids"],
            },
            {
                # LABELS
                # position of answer start token
                "start_positions": record["start_positions"],
                # position of answer end token
                "end_positions": record["end_positions"],
                # answer type which can range from 1 to 5
                "answer_types": record["answer_types"],
            },
        )

    x = {
        # FixedLenFeature used for parsing a fixed-length input feature
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
    # create batches of size batch size
    dataset = dataset.batch(batch_size) if batch_size != 0 else dataset
    #  map dataset to features dictionary for ease of access
    dataset = dataset.map(x_map)

    return dataset


def read_val_record(tf_record_file, shuffle_buffer_size, batch_size=1):
    """
    Reads tf records into a MapDataset for validation

    Args:
        tf_record_file: tf_record validation file
        shuffle_buffer_size: A chunk of this size of the dataset is shuffled for randomization
        batch_size: number of batches of consecutive data to be formed

    Returns: Validation dataset as a tuple with x(features)
    """

    def x_map(record):
        return {
            # FEATURES
            # unique_ids for every example
            "unique_ids": record["unique_ids"],
            # input_ids corresponding to tokens in the vocabulary
            "input_ids": record["input_ids"],
            # to make sure every input is same seq length
            "input_mask": record["input_mask"],
            # segment_ids to break input into different inputs
            "segment_ids": record["segment_ids"],
            # mapping from index of tokens of model input to index of tokens of original document text
            "token_map": record["token_map"],
        }

    x = {
        # FixedLenFeature used for parsing a fixed-length input feature
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
    # create batches of size batch size
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
        Args:
            inputs: inputs to the model in the form of input ids, input masks, and segment ids

        Returns: logits for start token, end token and answer type

        """
        inputs = inputs[:2] if isinstance(inputs, tuple) else inputs
        outputs = self.backend(inputs, **kwargs)
        # break distilbert output into sequence output and pooled output
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

        # For both short and long answers, we only generate one start and end token.
        # This is because we select the predicted long answer span as the
        # node containing the predicted
        # short answer span, and assign to both long and
        # short prediction the same score

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
    """
    Initializes optimizer (with schedule if necessary)

    Args:
        distilBert: model

    Returns: optimizer
    """
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # AdamW optimizer, slightly similar to what Google uses from tensorflow-addons library
    #     return tfa.optimizers.AdamW(weight_decay=FLAGS.init_weight_decay_rate,
    #                                       learning_rate=FLAGS.init_learning_rate,
    #                                   beta_1=0.9, beta_2=0.999,
    #                                   epsilon=1e-6)

    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # However, this doesn't cover learning-rate schedules, so I used a modification based on Yin dar shieh's version
    # on kaggle
    decay_steps = int(FLAGS.epochs * FLAGS.len_train / FLAGS.batch_size)

    # custom learning rate schedulers with warmup-steps and decay
    # source adam_optimizer.py
    # schedule = CustomSchedule(
    #     initial_learning_rate=FLAGS.init_learning_rate,
    #     decay_steps=decay_steps,
    #     end_learning_rate=FLAGS.init_learning_rate,
    #     power=1.0,
    #     cycle=True,
    #     num_warmup_steps=0,
    # )
    # I also tried a simple Polynomial Decay schedule, without warmup
    # source page https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay
    # schedule = PolynomialDecay(
    #     FLAGS.init_learning_rate,
    #     decay_steps,
    #     end_learning_rate=0.0001,
    #     power=1.0)

    # we want to run decay on all layers but "LayerNorm", "layer_norm", "bias".
    # However, there is no exclude parameter, so we are creating a "complement" list

    # decay_var_list = []
    #
    # for i in range(len(distilBert.trainable_variables)):
    #     name = distilBert.trainable_variables[i].name
    #     if any(x in name for x in ["LayerNorm", "layer_norm", "bias"]):
    #         # append everything but the 3 we dont want
    #         decay_var_list.append(name)

    # Modified AdamW optimizer, with learning rate schedule and warmup
    # source adam_optimizer.py
    # return AdamW(
    #     weight_decay=FLAGS.init_weight_decay_rate,
    #     learning_rate=schedule,
    #     beta_1=0.9,
    #     beta_2=0.999,
    #     epsilon=1e-6,
    #     decay_var_list=decay_var_list,
    # )

    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # eventually, I found a huggingface implementation that does pretty much what I was trying to implement above.
    # AdamW optimizer with a polynomial decay learning rate scheduler, with warm-up,
    # excluding "LayerNorm", "layer_norm", "bias" layers
    # source code https://huggingface.co/transformers/main_classes/optimizer_schedules.html
    return co(
        FLAGS.init_learning_rate, decay_steps, 1000, end_lr=0.0, optimizer_type="adamw"
    )

    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # LAMB optimizer, known for training BERT super fast (find it at https://arxiv.org/abs/1904.00962 )
    # Another optimizer, that is a modified version of AdamW specifically for BERT
    # source page https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/LAMB
    # return tfa.optimizers.LAMB(
    #     learning_rate=schedule,
    #     beta_1=0.9,
    #     beta_2=0.999,
    #     epsilon=1e-06,
    #     weight_decay_rate=FLAGS.init_weight_decay_rate,
    #     exclude_from_weight_decay = ["LayerNorm", "layer_norm", "bias"],
    #     name='LAMB'
    # )

    # -----------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------
# Find loss and gradients
# -----------------------------------------------------------------------------------------------------------------------


def compute_loss(labels, logits, depth):
    """
    Finds loss between logits and labels

    Args:
        labels: gold labels
        logits: generated predictions by model
        depth: number of possible answers, 5 for answer type, 512 for start and end token

    Returns:
        Loss value
    """

    # Implemented the way google defines loss in their bert-joint-baseline paper
    # For more information, check out model part of paper
    # convert labels to one hot
    one_hot_labels = tf.one_hot(labels, depth=depth, dtype=tf.float32)
    # find log probability of logits
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    # find loss by comparing labels and logit-probs
    loss = -tf.reduce_mean(tf.reduce_sum(one_hot_labels * log_probs, axis=-1))

    # using sparse categorical cross entropy
    # loss_sparse_cat = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = tf.math.reduce_sum(loss_sparse_cat(positions, logits))
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

    Args:
        (input_ids, input_masks, segment_ids): input features x
        (start_pos_labels, end_pos_labels, answer_type_labels): output labels y
        (train_acc, train_acc_start_pos, train_acc_end_pos, train_acc_ans_type): accuracy metrics

    Returns: Gradients, accuracy
    """
    # to speed up automatic differentiation, operations are recorded on a gradient 'tape'
    with tf.GradientTape() as tape:
        # find loss for all three outputs and average it to find total loss
        (start_pos_logits, end_pos_logits, answer_type_logits) = distilBert(
            (input_ids, input_masks, segment_ids), training=True
        )
        loss_start_pos = compute_loss(start_pos_labels, start_pos_logits, 512)
        loss_end_pos = compute_loss(end_pos_labels, end_pos_logits, 512)
        loss_ans_type = compute_loss(
            answer_type_labels, answer_type_logits, answer_types
        )
        total_loss = (loss_start_pos + loss_end_pos + loss_ans_type) / 3.0

    # compute gradient based on loss using auto differentiation
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

    Args:
         distilBert: model
         optimizer: optimizer for model
         train_dataset: training dataset
         ckpt_manager: checkpoint manager to store or read checkpoints
         (train_acc, train_acc_start_pos, train_acc_end_pos, train_acc_ans_type): accuracy metrics

    Returns: None but stores checkpoints as the training goes on
    """
    for epoch in range(FLAGS.epochs):
        # reset metrics at every epoch
        train_acc.reset_states()
        train_acc_start_pos.reset_states()
        train_acc_end_pos.reset_states()
        train_acc_ans_type.reset_states()

        for (instance, (x, y)) in enumerate(train_dataset):
            if FLAGS.clipped and instance == FLAGS.len_train:
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
                print("Epoch {}, Batches processed {}".format(epoch + 1, instance + 1,))

                print("Overall acc = {:.4f}".format(train_acc.result()))
                print("Start Token acc = {:.4f}".format(train_acc_start_pos.result()))
                print("End Token acc = {:.4f}".format(train_acc_end_pos.result()))
                print("Answer Type acc = {:.4f}".format(train_acc_ans_type.result()))

                print("-" * 100)

        if (epoch + 1) % 1 == 0:
            print(
                "\nSaving checkpoint for epoch {} at {}".format(
                    epoch + 1, ckpt_manager.save()
                )
            )

            print("Overall: loss = acc = {:.4f}".format(train_acc.result()))
            print("Start Token: acc = {:.4f}".format(train_acc_start_pos.result()))
            print("End Token: acc = {:.4f}".format(train_acc_end_pos.result()))
            print("Answer Type: acc = {:.4f}".format(train_acc_ans_type.result()))


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
    # retrieve pretrained model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    distilBert = TFNQModel.from_pretrained(model_name)
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
