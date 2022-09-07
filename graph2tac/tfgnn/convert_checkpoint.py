import re
import argparse
import tensorflow as tf
import tensorflow_gnn as tfgnn
from pathlib import Path

from graph2tac.tfgnn.predict import TFGNNPredict
from graph2tac.tfgnn.graph_schema import vectorized_definition_graph_spec, proofstate_graph_spec
from graph2tac.tfgnn.dataset import TFRecordDataset
from graph2tac.common import logger


def convert(log_dir: Path,
            checkpoint_number: int,
            proofstate_graph: tfgnn.GraphTensor,
            scalar_definition_graph: tfgnn.GraphTensor
            ) -> None:
    logger.info(f'converting checkpoint #{checkpoint_number} within {log_dir}')
    logger.warning(f'optimizer state will be lost; '
                   f'this should not matter for evaluation purposes, but will affect resumption of training')

    # load the prediction and definition tasks at the given checkpoint number
    predict = TFGNNPredict(log_dir=log_dir, checkpoint_number=checkpoint_number, numpy_output=False)

    # run some input through the prediction task once to make sure all weights are built and loaded
    # this is necessary to avoid problems with deferred checkpoint loading
    predict.prediction_task.prediction_model(proofstate_graph, training=False)

    # create a trainer checkpoint in the new format
    new_prediction_task_checkpoint = tf.train.Checkpoint(graph_embedding=predict.prediction_task.graph_embedding,
                                                         gnn=predict.prediction_task.gnn,
                                                         tactic_embedding=predict.prediction_task.tactic_embedding,
                                                         tactic_head=predict.prediction_task.tactic_head,
                                                         tactic_logits_from_embeddings=predict.prediction_task.tactic_logits_from_embeddings,
                                                         arguments_head=predict.prediction_task.arguments_head)

    new_trainer_checkpoint = tf.train.Checkpoint(prediction_task=new_prediction_task_checkpoint)

    if predict.definition_task is not None:
        # run some input through the definition task layer once to make sure all weights are built and loaded
        # this is necessary to avoid problems with deferred checkpoint loading
        predict.definition_task(scalar_definition_graph, training=False)

        # add the definition task to the trainer checkpoint in the new format
        new_trainer_checkpoint.definition_task = tf.train.Checkpoint(definition_head=predict.definition_task._definition_head)

    # try to add other trainer variables to the new checkpoint; none of these are actually necessary for evaluation
    old_checkpoint_path = log_dir / 'ckpt' / f'ckpt-{checkpoint_number}'
    checkpoint_reader = tf.train.load_checkpoint(str(old_checkpoint_path))
    try:
        definition_loss_coefficient = checkpoint_reader.get_tensor(
            'definition_loss_coefficient/.ATTRIBUTES/VARIABLE_VALUE')
    except tf.errors.NotFoundError:
        logger.warning(
            f'could not find definition_loss_coefficient within checkpoint #{checkpoint_number}; this should not matter for evaluation purposes, but will affect resumption of training')
    else:
        new_trainer_checkpoint.definition_loss_coefficient = tf.Variable(initial_value=definition_loss_coefficient,
                                                                         dtype=tf.float32, trainable=False)

    try:
        num_runs = checkpoint_reader.get_tensor('num_runs/.ATTRIBUTES/VARIABLE_VALUE')
    except tf.errors.NotFoundError:
        logger.warning(f'could not find num_runs within checkpoint #{checkpoint_number}; '
                       f'this should not matter for evaluation purposes, but will affect resumption of training')
    else:
        new_trainer_checkpoint.num_runs = tf.Variable(initial_value=num_runs, dtype=tf.int64, trainable=False)

    try:
        trained_epochs = checkpoint_reader.get_tensor('trained_epochs/.ATTRIBUTES/VARIABLE_VALUE')
    except tf.errors.NotFoundError:
        logger.warning(
            f'could not find trained_epochs within checkpoint #{checkpoint_number}; '
            f'this should not matter for evaluation purposes, but will affect resumption of training')
    else:
        new_trainer_checkpoint.trained_epochs = tf.Variable(initial_value=trained_epochs, dtype=tf.int64,
                                                            trainable=False)

    try:
        save_counter = checkpoint_reader.get_tensor('save_counter/.ATTRIBUTES/VARIABLE_VALUE')
    except tf.errors.NotFoundError:
        logger.warning(
            f'could not find save_counter within checkpoint #{checkpoint_number}; '
            f'this should not matter for evaluation purposes, but will affect resumption of training')
    else:
        new_trainer_checkpoint.save_counter.assign(save_counter)

    # save the new checkpoint in a new folder
    new_checkpoint_path = log_dir / 'ckpt_new' / f'ckpt-{checkpoint_number}'
    logger.info(f'saving checkpoint #{checkpoint_number} within {log_dir} to {new_checkpoint_path}')

    new_trainer_checkpoint.write(file_prefix=str(new_checkpoint_path))


def main():
    parser = argparse.ArgumentParser(description="Convert training checkpoints to the new (post-TFGNNPredict refactor) format")

    parser.add_argument("--log", type=Path, metavar="DIRECTORY",
                        help="Directory where checkpoints and logs are kept")

    args = parser.parse_args()

    sample_definitions_dataset = TFRecordDataset._parse_tfrecord_dataset(
        dataset=tf.data.TFRecordDataset(filenames=['sample_definitions.tfrecord']),
        graph_spec=vectorized_definition_graph_spec
    )
    scalar_definition_graph = sample_definitions_dataset.batch(3).get_single_element().merge_batch_to_components()

    sample_proofstate_dataset = TFRecordDataset._parse_tfrecord_dataset(
        dataset=tf.data.TFRecordDataset(filenames=['sample_proofstates.tfrecord']),
        graph_spec=proofstate_graph_spec
    )
    proofstate_graph = sample_proofstate_dataset.batch(3).get_single_element()

    old_checkpoints_path = args.log / 'ckpt'
    for ckpt in old_checkpoints_path.glob('*.index'):
        checkpoint_number = int(re.search('ckpt-(\d+).index', str(ckpt)).group(1))
        convert(log_dir=args.log,
                checkpoint_number=checkpoint_number,
                proofstate_graph=proofstate_graph,
                scalar_definition_graph=scalar_definition_graph)
    logger.info(f'all done, now you can delete the old {args.log}/ckpt and rename the new {args.log}/ckpt_new folder')


if __name__ == "__main__":
    main()
