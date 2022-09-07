import re
import argparse
import tensorflow as tf
from pathlib import Path

from graph2tac.tfgnn.predict import TFGNNPredict
from graph2tac.common import logger


def convert(log_dir: Path, checkpoint_number: int) -> None:
    logger.info(f'converting checkpoint #{checkpoint_number} within {log_dir}')
    logger.warning(f'optimizer state will be lost; '
                   f'this should not matter for evaluation purposes, but will affect resumption of training')

    predict = TFGNNPredict(log_dir=log_dir, checkpoint_number=checkpoint_number, numpy_output=False)

    new_prediction_task_checkpoint = tf.train.Checkpoint(graph_embedding=predict.prediction_task.graph_embedding,
                                                         gnn=predict.prediction_task.gnn,
                                                         tactic_embedding=predict.prediction_task.tactic_embedding,
                                                         tactic_head=predict.prediction_task.tactic_head,
                                                         tactic_logits_from_embeddings=predict.prediction_task.tactic_logits_from_embeddings,
                                                         arguments_head=predict.prediction_task.arguments_head)

    new_definition_task_checkpoint = tf.train.Checkpoint(definition_head=predict.definition_task._definition_head)

    new_trainer_checkpoint = tf.train.Checkpoint(prediction_task=new_prediction_task_checkpoint,
                                                 definition_task=new_definition_task_checkpoint)

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

    new_checkpoint_path = log_dir / 'ckpt_new' / f'ckpt-{checkpoint_number}'
    logger.info(f'saving checkpoint #{checkpoint_number} within {log_dir} to {new_checkpoint_path}')

    new_trainer_checkpoint.write(file_prefix=str(new_checkpoint_path))


def main():
    parser = argparse.ArgumentParser(description="Convert training checkpoints to the new (post-TFGNNPredict refactor) format")

    parser.add_argument("--log", type=Path, metavar="DIRECTORY",
                        help="Directory where checkpoints and logs are kept")

    args = parser.parse_args()

    old_checkpoints_path = args.log / 'ckpt'
    for ckpt in old_checkpoints_path.glob('*.index'):
        checkpoint_number = int(re.search('ckpt-(\d+).index', str(ckpt)).group(1))
        convert(log_dir=args.log, checkpoint_number=checkpoint_number)
    logger.info(f'all done, now you can delete the old {args.log}/ckpt and rename the new {args.log}/ckpt_new folder')


if __name__ == "__main__":
    main()
