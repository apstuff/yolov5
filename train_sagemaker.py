import typing
from pathlib import Path
import shutil

import environs
import sagemaker_ssh_helper

import train

if __name__ == "__main__":
    # Runs ssh helper (configured by estimator launch code)
    sagemaker_ssh_helper.setup_and_start_ssh()

    ############################################
    # Env
    ############################################

    env_args: typing.Dict[str, typing.Union[str, int, Path]] = {}
    env = environs.Env()
    # env.read_env() # read .env file, not needed in sagemaker env

    # Sagemaker env variables
    with env.prefixed("SM_"):
        # Directory where the dataset lives
        dataset_path = env.path("CHANNEL_YOLO_DATASET")
        model_dir = env.path("MODEL_DIR")
        output_data_dir = env.path("OUTPUT_DATA_DIR")

    # Yolov5 arguments
    with env.prefixed("YOLO_"):
        env_args["data"] = dataset_path / env.path("DATASET_FILE")
        env_args["hyp"] = dataset_path / env.path("HYP_FILE")
        # TODO maybe SAGEMAKER_JOB_NAME works across spot restarts?
        env_args["project"] = env.path("PROJECT")
        env_args["name"] = env.str("NAME")
        # overwrite any existing output
        env_args["exist_ok"] = True
        env_args["save_period"] = env.int("SAVE_PERIOD", -1)

        # only resume if the file exists
        resume_path = env.path("RESUME_PATH")
        if resume_path.exists():
            env_args['resume'] = str(resume_path)

    opt = train.parse_opt(True)
    for key, val in env_args.items():
        setattr(opt, key, val)

    train.main(opt)

    if (env_args['data'].parent/'images/test/').exists():
        import val as validate
        val_env_args : typing.Dict[str, typing.Union[str, int, Path]] = {}
        val_env_args['data'] = env_args['data']
        val_env_args['project'] = env_args['project']
        val_env_args['imgsz'] = opt.imgsz
        val_env_args['task'] = 'test'
        val_env_args['weights'] = str(resume_path.parent / 'best.pt')
        val_env_args['name'] = f"{env_args['name']}/test"
        opt = validate.parse_opt(True)
        for key, val in val_env_args.items():
            setattr(opt, key, val)
        validate.main(opt)

    src = resume_path.parent / 'best.pt'
    dst = model_dir
    print(f"Copy {src} to {dst}")
    shutil.copy2(src, dst)

    src = env_args["project"] / env_args['name']
    dst = output_data_dir
    print(f"Copy {src} to {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
