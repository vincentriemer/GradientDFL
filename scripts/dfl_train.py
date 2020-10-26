import sys
import argparse
import os
import time
import signal
import pexpect
import re
from gradient_statsd import Client

ITERATION_COUNT = 'iteration_count'
ITERATION_SPEED = 'iteration_speed'
SRC_LOSS = 'source_loss'
DST_LOSS = 'destination_loss'

TrainingIterationRe = r"\[\d\d:\d\d:\d\d\]\[#(\d+)\]\[(\d+)ms\]\[([\d.]+)\]\[([\d.]+)\]"
BooleanChoices = ["y", "n", ""]


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def run_training(autobackup_every_hour="",
                 write_preview_history="",
                 target_iteration="",
                 flip_faces_randomly="",
                 batch_size="",
                 masked_training="",
                 eyes_priority="",
                 uniform_yaw_distribution="",
                 place_models_on_gpu="",
                 learning_rate_dropout="",
                 random_warp="",
                 gan_power="",
                 face_style_power="",
                 bg_style_power="",
                 color_transfer="",
                 gradient_clipping="",
                 pretraining_mode=""):
    client = Client()

    cmd = "python /DeepFaceLab/main.py train"
    cmd += " --training-data-src-dir /storage/workspace/data_src/aligned"
    cmd += " --training-data-dst-dir /storage/workspace/data_dst/aligned"
    cmd += " --pretraining-data-dir /pretrain"
    cmd += " --model-dir /storage/workspace/model"
    cmd += " --model SAEHD"
    cmd += " --silent-start"

    child = pexpect.spawn(cmd, timeout=600)
    child.logfile_read = sys.stdout.buffer

    child.expect_exact("Press enter in 2 seconds to override model settings")
    child.sendline("")

    child.expect_exact("Autobackup every N hour ( 0..24 ?:help ) :")
    child.sendline(autobackup_every_hour)

    child.expect_exact("Write preview history ( y/n ?:help ) :")
    child.sendline(write_preview_history)

    child.expect_exact("Target iteration :")
    child.sendline(target_iteration)

    child.expect_exact("Flip faces randomly ( y/n ?:help ) :")
    child.sendline(flip_faces_randomly)

    child.expect_exact("Batch_size ( ?:help ) :")
    child.sendline(batch_size)

    child.expect_exact("Masked training ( y/n ?:help ) :")
    child.sendline(masked_training)

    child.expect_exact("Eyes priority ( y/n ?:help ) :")
    child.sendline(eyes_priority)

    child.expect_exact("Uniform yaw distribution of samples ( y/n ?:help ) :")
    child.sendline(uniform_yaw_distribution)

    child.expect_exact("Place models and optimizer on GPU ( y/n ?:help ) :")
    child.sendline(place_models_on_gpu)

    child.expect_exact("Use learning rate dropout ( n/y/cpu ?:help ) :")
    child.sendline(learning_rate_dropout)

    child.expect_exact("Enable random warp of samples ( y/n ?:help ) :")
    child.sendline(random_warp)

    child.expect_exact("GAN power ( 0.0 .. 10.0 ?:help ) :")
    child.sendline(gan_power)

    child.expect_exact("Face style power ( 0.0..100.0 ?:help ) :")
    child.sendline(face_style_power)

    child.expect_exact("Background style power ( 0.0..100.0 ?:help ) :")
    child.sendline(bg_style_power)

    child.expect_exact(
        "Color transfer for src faceset ( none/rct/lct/mkl/idt/sot ?:help ) :")
    child.sendline(color_transfer)

    child.expect_exact("Enable gradient clipping ( y/n ?:help ) :")
    child.sendline(gradient_clipping)

    child.expect_exact("Enable pretraining mode ( y/n ?:help ) :")
    child.sendline(pretraining_mode)

    # child.expect_exact("Initializing models:")
    # child.expect_exact("Loading samples:")
    # idx = child.expect_exact(["Loading samples:", "Sort by yaw:"])
    # if idx == 1:
    #   child.expect_exact("Loading samples:")
    #   child.expect_exact("Sort by yaw:")
    child.expect_exact(
        'Starting. Press "Enter" to stop training and save model.')

    child.logfile_read = None

    Done = False
    while not Done:
        child.expect(TrainingIterationRe)

        iter_string = child.after.decode("utf-8")
        match = re.search(TrainingIterationRe, iter_string)

        current_iteration = num(match.group(1))
        iteration_time = num(match.group(2))
        src_loss_value = num(match.group(3))
        dst_loss_value = num(match.group(4))

        client.gauge(ITERATION_COUNT, current_iteration)
        client.gauge(ITERATION_SPEED, iteration_time)
        client.gauge(SRC_LOSS, src_loss_value)
        client.gauge(DST_LOSS, dst_loss_value)

        print('[#{}][{}ms][{}][{}]'.format(
            current_iteration, iteration_time, src_loss_value, dst_loss_value), end="\n")

        if current_iteration % 1000 == 0:
            os.system('zip -r -q /artifacts/model.zip /storage/workspace/model')
            print('Updated Model Artifact!')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--autobackup-every-hour", default="",
                        dest="autobackup_every_hour", help="Autobackup every N hour (0..24)")
    parser.add_argument("--write-preview_history", default="",
                        choices=BooleanChoices, dest="write_preview_history")
    parser.add_argument("--target-iteration", default="",
                        dest="target_iteration")
    parser.add_argument("--flip-faces-randomly", default="",
                        choices=BooleanChoices, dest="flip_faces_randomly")
    parser.add_argument("--batch-size", default="", dest="batch_size")
    parser.add_argument("--masked-training", default="",
                        choices=BooleanChoices, dest="masked_training")
    parser.add_argument("--eyes-priority", default="",
                        choices=BooleanChoices, dest="eyes_priority")
    parser.add_argument("--uniform-yaw-distribution", default="",
                        choices=BooleanChoices, dest="uniform_yaw_distribution")
    parser.add_argument("--place-models-on-gpu", default="",
                        choices=BooleanChoices, dest="place_models_on_gpu")
    parser.add_argument("--learning-rate-dropout", default="",
                        choices=BooleanChoices, dest="learning_rate_dropout")
    parser.add_argument("--random-warp", default="",
                        choices=BooleanChoices, dest="random_warp")
    parser.add_argument("--gan-power", default="", dest="gan_power")
    parser.add_argument("--face-style-power", default="",
                        dest="face_style_power")
    parser.add_argument("--bg-style-power", default="", dest="bg_style_power")
    parser.add_argument("--color-transfer", default="", dest="color_transfer")
    parser.add_argument("--gradient-clipping", default="",
                        choices=BooleanChoices, dest="gradient_clipping")
    parser.add_argument("--pretraining-mode", default="",
                        choices=BooleanChoices, dest="pretraining_mode")

    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == '__main__':
    main()
