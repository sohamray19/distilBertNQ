# -----------------------------------------------------------------------------------------------------------------------------------------------
# Code referenced from Google bert-joint-baseline to generate "predictions.json" for evaluation script

import collections
import json
import gzip
import tensorflow as tf

max_answer_length = 30
best_indexes_size = 20

Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])


class EvalExample(object):
    """Eval data available for a single example."""

    def __init__(self, example_id, candidates):
        self.example_id = example_id
        self.candidates = candidates
        self.results = {}
        self.features = {}


class ScoreSummary(object):
    def __init__(self):
        self.predicted_label = None
        self.short_span_score = None
        self.cls_token_score = None
        self.answer_type_logits = None
        self.start_prob = None
        self.end_prob = None
        self.answer_type_prob_dist = None


def read_candidates_from_one_split(input_path):
    """Read candidates from a single jsonl file."""
    candidates_dict = {}
    if input_path.endswith(".gz"):
        with gzip.GzipFile(fileobj=tf.io.gfile.GFile(input_path, "rb")) as input_file:
            print("Reading examples from: {}".format(input_path))
            for index, line in enumerate(input_file):
                e = json.loads(line)
                candidates_dict[e["example_id"]] = e["long_answer_candidates"]
                # if index > 100:
                #     break
    else:
        with tf.io.gfile.GFile(input_path, "r") as input_file:
            print("Reading examples from: {}".format(input_path))
            for index, line in enumerate(input_file):
                e = json.loads(line)
                candidates_dict[e["example_id"]] = e["long_answer_candidates"]
                # if index > 100:
                #     break

    return candidates_dict


def read_candidates(input_pattern):
    """Read candidates with real multiple processes."""
    input_paths = tf.io.gfile.glob(input_pattern)
    final_dict = {}
    for input_path in input_paths:
        final_dict.update(read_candidates_from_one_split(input_path))
    return final_dict


def get_best_indexes(logits, best_indexes_size, token_map=None):
    # Return a sorted list of (idx, logit)
    index_and_score = sorted(enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):

        idx = index_and_score[i][0]

        if token_map is not None and token_map[idx] == -1:
            continue

        best_indexes.append(idx)

        if len(best_indexes) >= best_indexes_size:
            break

    return best_indexes


def compute_predictions(example):
    """Converts an example into an NQEval object for evaluation.

       Unlike the starter kernel, this returns a list of `ScoreSummary`, sorted by score.
    """

    predictions = []

    for unique_id, result in example.results.items():

        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = example.features[unique_id]["token_map"].int64_list.value

        for start_index, start_logit, start_prob in zip(
            result["start_indexes"],
            result["start_logits"],
            result["start_pos_prob_dist"],
        ):

            if token_map[start_index] == -1:
                continue
            for end_index, end_logit, end_prob in zip(
                result["end_indexes"], result["end_logits"], result["end_pos_prob_dist"]
            ):

                if token_map[end_index] == -1:
                    continue

                if end_index < start_index:
                    continue

                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue

                summary = ScoreSummary()

                summary.short_span_score = start_logit + end_logit
                summary.cls_token_score = (
                    result["cls_start_logit"] + result["cls_end_logit"]
                )
                summary.answer_type_logits = result["answer_type_logits"]

                summary.start_indexes = result["start_indexes"]
                summary.end_indexes = result["end_indexes"]

                summary.start_logits = result["start_logits"]
                summary.end_logits = result["end_logits"]

                summary.start_pos_prob_dist = result["start_pos_prob_dist"]
                summary.end_pos_prob_dist = result["end_pos_prob_dist"]

                summary.start_index = start_index
                summary.end_index = end_index

                summary.start_logit = start_logit
                summary.end_logit = end_logit

                answer_type_prob_dist = result["answer_type_prob_dist"]
                summary.start_prob = start_prob
                summary.end_prob = end_prob
                summary.answer_type_prob_dist = {
                    "unknown": answer_type_prob_dist[0],
                    "yes": answer_type_prob_dist[1],
                    "no": answer_type_prob_dist[2],
                    "short": answer_type_prob_dist[3],
                    "long": answer_type_prob_dist[4],
                }
                start_span = token_map[start_index]
                end_span = token_map[end_index] + 1

                # Span logits minus the cls logits seems to be close to the best.
                score = summary.short_span_score - summary.cls_token_score
                predictions.append((score, summary, start_span, end_span))

    all_summaries = []

    if predictions:

        predictions = sorted(
            predictions, key=lambda x: (x[0], x[2], x[3]), reverse=True
        )

        for prediction in predictions:

            long_span = Span(-1, -1)

            score, summary, start_span, end_span = prediction
            short_span = Span(start_span, end_span)
            for c in example.candidates:
                start = short_span.start_token_idx
                end = short_span.end_token_idx
                if (
                    c["top_level"]
                    and c["start_token"] <= start
                    and c["end_token"] >= end
                ):
                    long_span = Span(c["start_token"], c["end_token"])
                    break

            summary.predicted_label = {
                "example_id": example.example_id,
                #                 "instance_id": example.instance_id,
                "long_answer": {
                    "start_token": long_span.start_token_idx,
                    "end_token": long_span.end_token_idx,
                    "start_byte": -1,
                    "end_byte": -1,
                },
                "short_answers": [
                    {
                        "start_token": short_span.start_token_idx,
                        "end_token": short_span.end_token_idx,
                        "start_byte": -1,
                        "end_byte": -1,
                    }
                ],
                "yes_no_answer": "NONE",
                "long_answer_score": score,
                "short_answers_score": score,
            }
            all_summaries.append(summary)

    if len(all_summaries) == 0:
        short_span = Span(-1, -1)
        long_span = Span(-1, -1)
        score = 0
        summary = ScoreSummary()

        summary.predicted_label = {
            "example_id": example.example_id,
            #                 "instance_id": None,
            "long_answer": {
                "start_token": long_span.start_token_idx,
                "end_token": long_span.end_token_idx,
                "start_byte": -1,
                "end_byte": -1,
            },
            "long_answer_score": score,
            "short_answers": [
                {
                    "start_token": short_span.start_token_idx,
                    "end_token": short_span.end_token_idx,
                    "start_byte": -1,
                    "end_byte": -1,
                }
            ],
            "short_answers_score": score,
            "yes_no_answer": "NONE",
        }

        all_summaries.append(summary)
    all_summaries = all_summaries[: min(best_indexes_size, len(all_summaries))]

    return all_summaries


def compute_pred_dict(candidates_dict, dev_features, raw_results):
    """Computes official answer key from raw logits.

       Unlike the starter kernel, each nq_pred_dict[example_id] is a list of `predicted_label`
       that is defined in `compute_predictions`.
    """

    raw_results_by_id = [(int(res["unique_id"]), 1, res, None) for res in raw_results]
    examples_by_id = [
        (int(tf.cast(int(k), dtype=tf.int32)), 0, v, k)
        for k, v in candidates_dict.items()
    ]

    features_by_id = [
        (
            int(
                tf.cast(
                    f.features.feature["unique_ids"].int64_list.value[0], dtype=tf.int32
                )
            ),
            2,
            f.features.feature,
            None,
        )
        for f in dev_features
    ]

    print("merging examples...")
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
    print("done.")
    print("len merged:", len(merged))
    examples = []
    for idx, type_, datum, orig_example_id in merged:
        if type_ == 0:  # Here, datum the list `long_answer_candidates`
            examples.append(EvalExample(orig_example_id, datum))
        elif type_ == 2:  # Here, datum is a feature with `token_map`
            examples[-1].features[idx] = datum
        else:  # Here, datum is a raw_result given by the model
            examples[-1].results[idx] = datum

            # Construct prediction objects.
    summary_dict = {}
    nq_pred_dict = {}
    for e in examples:

        all_summaries = compute_predictions(e)
        summary_dict[e.example_id] = all_summaries
        nq_pred_dict[e.example_id] = [
            summary.predicted_label for summary in all_summaries
        ]
        if len(nq_pred_dict) % 100 == 0:
            print("Examples processed: %d" % len(nq_pred_dict))

    return nq_pred_dict


def get_prediction_json(
    distilBert, val_dataset, pred_file, val_file, json_output_path, best_indexes_size=-1
):
    dataset = val_dataset
    eval_features = (
        tf.train.Example.FromString(r.numpy())
        for r in tf.data.TFRecordDataset(val_file)
    )

    print(pred_file)
    print(json_output_path)

    all_results = []

    for (batch_idx, features) in enumerate(dataset):

        unique_ids = features["unique_ids"]
        token_maps = features["token_map"]

        (input_ids, input_masks, segment_ids) = (
            features["input_ids"],
            features["input_mask"],
            features["segment_ids"],
        )

        nq_inputs = (input_ids, input_masks, segment_ids)
        nq_logits = distilBert(nq_inputs, training=False)

        (start_pos_logits, end_pos_logits, answer_type_logits) = nq_logits

        unique_ids = unique_ids.numpy().tolist()

        token_maps = token_maps.numpy().tolist()

        start_pos_prob_dist = tf.nn.softmax(start_pos_logits, axis=-1).numpy().tolist()
        end_pos_prob_dist = tf.nn.softmax(end_pos_logits, axis=-1).numpy().tolist()
        answer_type_prob_dist = (
            tf.nn.softmax(answer_type_logits, axis=-1).numpy().tolist()
        )

        start_pos_logits = start_pos_logits.numpy().tolist()
        end_pos_logits = end_pos_logits.numpy().tolist()
        answer_type_logits = answer_type_logits.numpy().tolist()

        for uid, token_map, s, e, a, sp, ep, ap in zip(
            unique_ids,
            token_maps,
            start_pos_logits,
            end_pos_logits,
            answer_type_logits,
            start_pos_prob_dist,
            end_pos_prob_dist,
            answer_type_prob_dist,
        ):
            if best_indexes_size < 0:
                best_indexes_size = len(start_pos_logits)

            cls_start_logit = s[0]
            cls_end_logit = e[0]

            start_indexes = get_best_indexes(s, best_indexes_size, token_map)
            end_indexes = get_best_indexes(e, best_indexes_size, token_map)

            s = [s[idx] for idx in start_indexes]
            e = [e[idx] for idx in end_indexes]
            sp = [sp[idx] for idx in start_indexes]
            ep = [ep[idx] for idx in end_indexes]

            raw_result = {
                "unique_id": uid,
                "start_indexes": start_indexes,
                "end_indexes": end_indexes,
                "start_logits": s,
                "end_logits": e,
                "answer_type_logits": a,
                "start_pos_prob_dist": sp,
                "end_pos_prob_dist": ep,
                "answer_type_prob_dist": ap,
                "cls_start_logit": cls_start_logit,
                "cls_end_logit": cls_end_logit,
            }
            all_results.append(raw_result)

        if (batch_idx + 1) % 100 == 0:
            print("Batch {} processed".format(batch_idx + 1))

    print("Going to candidates file")
    candidates_dict = read_candidates(pred_file)

    print("compute_pred_dict")
    nq_pred_dict = compute_pred_dict(candidates_dict, eval_features, all_results)

    print(nq_pred_dict.values)
    predictions_json = {"predictions": list(v[0] for v in nq_pred_dict.values())}

    print("writing json")
    with tf.io.gfile.GFile(json_output_path, "w") as f:
        json.dump(predictions_json, f, indent=4)


# -------------------------------------------------------------------------------------------------------------------------------------------------
