import os
import argparse

import yolov5.train
import environs
import sagemaker_ssh_helper

if __name__ == "__main__":
    # Runs ssh helper (configured by estimator launch code)
    sagemaker_ssh_helper.setup_and_start_ssh()

    ############################################
    # Env
    ############################################

    env_args: dict[str, str] = {}
    env = environs.Env()
    env.read_env()

    # Sagemaker env variables
    with env.prefixed("SM_"):
        env_args["model_dir"] = env.path("MODEL_DIR", "/opt/ml/model/")
        env_args["output_dir"] = env.path("OUTPUT_DIR", "/opt/ml/output/")
        env_args["checkpoint_dir"] = env.path("CHECKPOINT_DIR", "/opt/ml/checkpoints/")

    # Yolov5 arguments
    with env.prefixed("YOLO_"):
        env_args["imgsz"] = env.int("IMGSZ", 512)
        env_args["batch_size"] = env.int("BATCH", 32)
        env_args["weights"] = env.str("WEIGHTS")
        env_args["data"] = env.path("DATA")
        env_args["hyp"] = env.path("HYP")
        env_args["project"] = env.path("PROJECT", "/opt/ml/runs/")
        env_args["name"] = env.str("NAME")
        env_args["save_period"] = env.int("SAVE_PERIOD", -1)

    opt = yolov5.train.parse_opt(True)
    for key, val in env_args.items():
        setattr(opt, key, val)

    print(opt)
    yolov5.train.main(opt)
