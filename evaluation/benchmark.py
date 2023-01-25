import argparse
import shutil
import os
from pathlib import Path
from subprocess import run
from time import time

import numpy as np
from tqdm import tqdm

EVAL_DIR = Path(__file__).parent
LFW_THRESHOLD = 0.3


def benchmark_accuracy_and_latency(
    model_path: Path, start_index: int, n_samples: int, model_name: str
) -> None:
    t = []
    for j in tqdm(range(start_index, start_index + n_samples), "Inference"):
        t0 = time()
        os.system(
                " ".join(
                [
                    "../target/release/he-man-concrete",
                    "inference",
                    "container/",
                    str(model_path),
                    f"container/{j}.enc",
                    f"container/{j}.out.enc",
                ]
            )
        )
        t1 = time()
        t.append(t1 - t0)

    for j in tqdm(range(start_index, start_index + n_samples), "Decryption"):
        os.system(
            " ".join(
                [
                    "../target/release/he-man-concrete",
                    "decrypt",
                    "container/",
                    f"container/{j}.out.enc",
                    f"container/{j}.out.npy",
                ]
            )
        )

    if model_name == "mobilefacenets":
        correct_labels = np.array([True, False] * 500)
        predictions = []

        for j in tqdm(range(start_index, start_index + n_samples), "Evaluation"):
            left = np.load(f"container/{j}.out.npy").squeeze()
            right = np.load(f"lfw_verification/{j}.npy").squeeze()
            left = left / np.sqrt(np.sum(np.power(left, 2)))
            right = right / np.sqrt(np.sum(np.power(right, 2)))
            score = np.dot(left, right)
            predictions.append(score > LFW_THRESHOLD)
    else:
        correct_labels = np.load("mnist_data/labels.npy")
        predictions = []

        for j in tqdm(range(start_index, start_index + n_samples), "Evaluation"):
            label_pred = np.load(f"container/{j}.out.npy")
            predictions.append(np.argmax(label_pred))

    end_index = start_index + n_samples
    accuracy = (
        (np.array(predictions) == correct_labels[start_index:end_index])
        .astype(float)
        .mean()
    )

    print(f"accuracy: {accuracy:.4f}")
    print(f"mean inference time: {np.array(t).mean():.4f}s")


def prepare_calibrate(
    model: Path, calibration_data_file: Path, calibrated_model_path: Path
) -> None:
    print("Generate keyparams ...", end=" ")

    # check if model with name of calibration model exists and delete model
    if calibrated_model_path.exists():
        calibrated_model_path.unlink()

    os.system(
        " ".join(
            [
                "../target/release/he-man-concrete",
                "keyparams",
                str(model),
                str(calibration_data_file),
                "container/keyparams.json",
                str(calibrated_model_path),
            ]
        )
    )
    print("Done")


def prepare_generate_keys() -> None:
    print("Generate keys ...", end=" ")
    os.system(
        " ".join(
            [
                "../target/release/he-man-concrete",
                "keygen",
                "container/keyparams.json",
                "container/",
            ]
        )
    )
    print("Done")


def prepare_encrypt(input_data_dir: Path, start_index: int, n_samples: int) -> None:
    for j in tqdm(range(start_index, start_index + n_samples), "Encrypt"):
        os.system(
            " ".join(
                [
                    "../target/release/he-man-concrete",
                    "encrypt",
                    "container/",
                    str(input_data_dir / f"{j}.npy"),
                    f"container/{j}.enc",
                ]
            )
        )


def check_and_prepare_evaluation_dir(
    model_path: Path, input_data_dir: Path, calibration_data_file: Path
) -> None:
    print("Pre-Evaluation Checks ...", end=" ")
    # check if model exists
    if not model_path.exists():
        print(f"model {model_path.name} does not exist. abort.")
        exit()

    if not input_data_dir.exists():
        print("input data directory missing. abort.")
        exit()

    if not all((input_data_dir / f"{j}.npy").exists() for j in range(1000)):
        print("input data missing. abort.")
        exit()

    # check if input data exist
    if args.model in ["cryptonets", "lenet5"]:
        if not (input_data_dir / "labels.npy").exists():
            print("mnist labels missing. abort.")
            exit()

    # check if calibration data file exists
    if not calibration_data_file.exists():
        print("calibration data missing. abort.")
        exit()

    # delete container content
    container_path = Path(__file__).parent / "container"
    if not container_path.exists():
        print("container directory is missing. abort.")
        exit()

    elif container_path.is_dir():
        for f in container_path.iterdir():
            if f.is_file() and f.name != ".gitignore":
                f.unlink()

    print("Successful")


def perform_benchmark(
    model_path: Path,
    input_data_dir: Path,
    calibration_data_file: Path,
    start_index: int,
    n_samples: int,
    model_name: str,
) -> None:
    calibrated_model_path = model_path.parent / (
        model_path.stem + "_calibrated" + model_path.suffix
    )

    prepare_calibrate(model_path, calibration_data_file, calibrated_model_path)
    prepare_generate_keys()
    prepare_encrypt(input_data_dir, start_index, n_samples)
    benchmark_accuracy_and_latency(
        calibrated_model_path, start_index, n_samples, model_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HE-MAN evaluation")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="lenet5",
        choices=["cryptonets", "lenet5", "mobilefacenets"],
        help="model to be executed for benchmark",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="start index of dataset",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=10,
        help="number of samples to execute",
    )
    args = parser.parse_args()

    model_path = EVAL_DIR / (args.model + ".onnx")

    if args.model == "mobilefacenets":
        # check if zip file exists, if not -> zip the calibration data
        calibration_data_path = EVAL_DIR / "lfw_calibration-data.zip"
        if not calibration_data_path.exists():
            if (EVAL_DIR / "lfw_calibration-data").exists():
                shutil.make_archive(
                    str(calibration_data_path.parent / calibration_data_path.stem),
                    "zip",
                    root_dir="lfw_calibration-data/",
                    base_dir="./",
                )
            else:
                print("No LFW calibration data found. abort.")
                exit()

        input_data_dir = EVAL_DIR / "lfw_data"
    else:
        input_data_dir = EVAL_DIR / "mnist_data"

    if args.model == "mobilefacenets":
        calibration_data_file = EVAL_DIR / "lfw_calibration-data.zip"
    else:
        calibration_data_file = EVAL_DIR / "mnist_calibration-data.zip"

    check_and_prepare_evaluation_dir(model_path, input_data_dir, calibration_data_file)

    perform_benchmark(
        model_path,
        input_data_dir,
        calibration_data_file,
        args.start,
        args.number,
        args.model,
    )
