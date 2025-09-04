import os
import argparse
from huggingface_hub import HfApi, HfHubHTTPError, RepositoryNotFoundError


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Download SpatialVID dataset from Hugging Face Hub.")
    parser.add_argument(
        "--type",
        type=str,
        choices=["SpatialVID", "SpatialVID-HQ"],
        default="SpatialVID",
        help="Dataset type (default: SpatialVID)"
    )
    parser.add_argument(
        "--hub_path",
        type=str,
        help="Target path in Hugging Face repo (e.g., 'videos/' or 'metadata.csv'). "
             "Omit to download entire repo.",
        default=None
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Local directory to save dataset",
        default="/path/to/dataset",
    )
    args = parser.parse_args()

    # Initialize API and parameters
    api = HfApi()
    repo_id = f"SpatialVID/{args.type}"
    local_dir = args.output_dir

    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
            print(f"Created output directory: {local_dir}")

        # Download based on hub_path
        if args.hub_path is not None:
            # Try downloading as folder first, then file if that fails
            try:
                api.download_folder(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=args.hub_path,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    progress_bar=True
                )
                print(f"Downloaded directory '{args.hub_path}' from {repo_id} to {local_dir}")
            except HfHubHTTPError as e:
                if "not a directory" in str(e).lower():
                    api.download_file(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=args.hub_path,
                        local_dir=local_dir
                    )
                    print(f"Downloaded file '{args.hub_path}' from {repo_id} to {local_dir}")
                else:
                    raise
        else:
            # Download entire repository
            api.download_repo(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print(f"Downloaded entire repo {repo_id} to {local_dir}")

    # Error handling
    except RepositoryNotFoundError:
        print(f"Error: Repository '{repo_id}' not found. Check repo ID.")
    except HfHubHTTPError as e:
        print(f"HTTP Error: {str(e)}. Possible issues: network, invalid path, or permissions.")
    except OSError as e:
        print(f"File System Error: {str(e)}. Check output path or permissions.")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")


if __name__ == "__main__":
    main()
