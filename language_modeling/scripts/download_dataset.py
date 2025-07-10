import argparse
from pathlib import Path
from datasets import load_dataset

def download_and_save_dataset(dataset_name, save_dir, split, filename, branch=None):
    """
    Downloads a dataset from Hugging Face datasets library and saves it to a specified path.

    Args:
        dataset_name (str): The name of the dataset to download (e.g., "imdb").
        save_dir (str): The directory where the dataset will be saved.
        split (str): The data split to download (e.g., "train", "test").
        branch (str or None): The branch of the dataset to load (can be None for default).
    """
    # Load the dataset with the given split and branch (if provided)
    dataset = load_dataset(dataset_name, name=branch, split=split)

    # Save the dataset
    save_path = Path(save_dir) / dataset_name / f"{filename}.json"
    dataset.to_json(save_path)
    print(f"Dataset '{dataset_name}' ({split} split) has been saved to '{save_path}'.")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Download and save a Hugging Face dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to download.")
    parser.add_argument("--save_dir", type=str, required=True, help="The directory to save the dataset.")
    parser.add_argument("--split", type=str, required=True, help="The data split to download (e.g., 'train', 'test').")
    parser.add_argument("--filename", type=str, required=True, help="The data split to download (e.g., 'train', 'test').")
    parser.add_argument("--branch", type=str, default=None, help="The branch of the dataset to load (default: None).")

    # Parse arguments
    args = parser.parse_args()

    # Download and save the dataset
    download_and_save_dataset(args.dataset_name, args.save_dir, args.split, args.filename, args.branch)
