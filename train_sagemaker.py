import typing
import shutil
from pathlib import Path
import yaml

import environs
import sagemaker_ssh_helper

import train
import val as validate
import export


def main():
    """
    Wraps yolo's train.py to be called from sagemaker.Estimator

    Assumes the following directory structure, which maps to paths in S3

    /yolov5                      - s3://{bucket}/model/{base_job_name}/checkpoints/{uuid}
    ├── train
    └── test
    /opt/ml
    ├── code                     - s3://{bucket}/model/{base_job_name}/code/{job_name}/source.tar.gz
    ├── input
        └── data
            ├── weights
            └── yolo_dataset     - s3://{bucket}/datasets/{dataset_name}
    ├── model                    - s3://{bucket}/model/{base_job_name}/output/{job_name}/output/model.tar.gz
    └── output
        └── data                 - s3://{bucket}/model/{base_job_name}/output/{job_name}/output/output.tar.gz
            ├── train
            └── test

    /yolov5/train: yolo save_dir for train.py (ie {project}/{name}). Will be continously synced to S3.
    /yolov5/test: yolo save_dir for val.py.
    /opt/ml/code: This is the working directory for the sagemaker container.
    /opt/ml/input/data/yolo_dataset: This is where Sagemaker puts data from S3.
    /opt/ml/input/data/weights: If pretrained weights are provided from S3 they will be mounted here.
    /opt/ml/model: At the end of training, best.pt is copied here so that Sagemaker can export it.
    /opt/ml/output/data: At the end of training and testing the respective save_dir contents are copied here for Sagemaker to export.

    Also see: https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage.html

    There are two kinds of arguments passed to this script by Estimator, args and env variables (prefixed with YOLO_).
    Args are meant to be changing often, such as number of epochs or batch size.
    Env variables are meant to change rarely, such as the name of the dataset file.

    Sagemaker also passes its own information through env variables (prefixed with SM_).

    Env args and args are merged into one object and passed to yolo training.

    If a test dataset is detected the script will also perform a test run (using val.py).

    Finally all artifacts are copied to where Sagemaker expects them to be, such that they can be persisted.
    """

    # Runs ssh helper (configured by estimator launch code)
    sagemaker_ssh_helper.setup_and_start_ssh()

    """
    Env
    """

    env = environs.Env()
    # env.read_env() # read .env file, not needed in sagemaker env

    # Sagemaker env variables
    with env.prefixed("SM_"):
        # Directory where the dataset lives
        dataset_path: Path = env.path("CHANNEL_YOLO_DATASET")
        model_dir: Path = env.path("MODEL_DIR")
        output_data_dir: Path = env.path("OUTPUT_DATA_DIR")

        weights_path: typing.Optional[Path] = env.path("CHANNEL_WEIGHTS", None)

    # Yolov5 arguments
    with env.prefixed("YOLO_"):
        dataset_file = dataset_path / env.path("DATASET_FILE")
        hyp_file = dataset_path / env.path("HYP_FILE")
        # overwrite any existing output
        save_period = env.int("SAVE_PERIOD", -1)

        save_dir: Path = env.path("SAVE_DIR")
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        assert save_dir.is_dir()

    """
    Train
    """

    train_opt = train.parse_opt(known=True)
    train_args = {
        "data": str(dataset_file),
        "hyp": str(hyp_file),
        "save_period": save_period,
        "project": str(save_dir),
        "name": "train",
        "exist_ok": True,
    }
    # only resume if the file exists
    resume_path: Path = (
        Path(train_args["project"]) / train_args["name"] / "weights" / "last.pt"
    )
    if resume_path.exists():
        train_args["resume"] = str(resume_path)
    for key, val in train_args.items():
        setattr(train_opt, key, val)

    # special case: use pretrained weights from S3
    assert (train_opt.weights.startswith("s3://") and weights_path is not None) or (
        weights_path is None
    )
    if train_opt.weights.startswith("s3://"):
        print("Detected pretrained weights")
        assert weights_path is not None and weights_path.exists()
        new_weights = str(weights_path / Path(train_opt.weights).name)
        setattr(train_opt, "weights", new_weights)

    print("train_opt:", str(train_opt))
    train.main(train_opt)

    """
    Export
    """

    # export model to /yolov5/train/weights/best.torchscript
    export_opt = export.parse_opt(known=True)
    export_args = {
        "weights": str(resume_path.parent / "best.pt"),
        "include": ["torchscript"],
    }
    for key, val in export_args.items():
        setattr(export_opt, key, val)

    print("export_opt", str(export_opt))
    export.main(export_opt)

    # copy model to /opt/ml/model
    src = resume_path.parent / "best.torchscript"
    dst = model_dir
    print(f"Copy model {src} to {dst}")
    shutil.copy2(src, dst)

    # copy train outputs to /opt/ml/output/data/train
    src = Path(train_args["project"]) / train_args["name"]
    dst = output_data_dir / train_args["name"]
    print(f"Copy training outputs {src} to {dst}")
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("*.sagemaker-upload*"),
        dirs_exist_ok=True,
    )

    """
    Test
    """

    with open(train_args["data"], "r") as file:
        dataset = yaml.safe_load(file)

    if not ("test" in dataset and (Path(dataset["path"]) / dataset["test"]).exists()):
        return

    test_opt = validate.parse_opt(known=True)
    test_args = {
        "data": train_opt.data,
        "project": train_opt.project,
        "name": "test",
        "exist_ok": True,
        "imgsz": train_opt.imgsz,
        "task": "test",
        "weights": str(
            Path(train_opt.project) / train_opt.name / "weights" / "best.pt"
        ),
    }
    for key, val in test_args.items():
        setattr(test_opt, key, val)

    print("test_opt:", str(test_opt))
    validate.main(test_opt)

    # copy test outputs to /opt/ml/output/data/train
    src = Path(test_args["project"]) / test_args["name"]
    dst = output_data_dir / test_args["name"]
    print(f"Copy test outputs {src} to {dst}")
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("*.sagemaker-upload*"),
        dirs_exist_ok=True,
    )


if __name__ == "__main__":
    main()
