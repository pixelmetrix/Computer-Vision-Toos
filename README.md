# Computer-Vision-Tools

A Python library for preprocessing high-resolution images into smaller, normalized patches for computer vision model training.

## Overview

This library processes large high-resolution images (e.g., 6000x8000 pixels) by:
- Cutting them into smaller 1024x1024 pixel patches
- Organizing the processed images into structured folders
- Preparing normalized datasets for training computer vision models

The main workflow:
1. Place high-resolution images in folders within `Images/Raw_imgs/`
2. Create a `Normalization()` instance pointing to a specific folder
3. Call the `.preprocess()` method to automatically cut and save image patches
4. Processed patches are saved in `Images/Processed_imgs/original_folder_name_processed/`

## Requirements

This project uses [Poetry](https://python-poetry.org/) for dependency management. You must have Poetry installed on your system before proceeding.

### Installing Poetry

If you don't have Poetry installed, follow the [official installation guide](https://python-poetry.org/docs/#installation):

```bash
# Linux, macOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -

# Or via pip
pip install poetry
```

For more details, see the [Poetry documentation](https://python-poetry.org/docs/basic-usage/).

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Computer-Vision-Tools
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```
   
   This command will:
   - Read the `poetry.lock` file to ensure exact dependency versions
   - Create a virtual environment automatically
   - Install all required dependencies
   - Install the project in editable mode

3. **Verify installation**
   ```bash
   poetry show
   ```
   
   This will list all installed packages.

## Usage

### Running the normalization module

You can run the normalization module using `poetry run`:

```bash
poetry run python -m modules.normalization
```

### Activating the virtual environment

Alternatively, you can activate the Poetry shell and run commands directly:

```bash
poetry shell
python -m modules.normalization
```

### Using the Normalization class

```python
from modules.normalization import Normalization

# Create an instance pointing to a folder in Raw_imgs/
normalizer = Normalization(folder_name="my_images")

# Process the high-resolution images into 1024x1024 patches
normalizer.preprocess()

# Processed images will be saved in:
# Images/Processed_imgs/my_images_processed/
```

### Deleting processed images

The `Normalization` class includes a `delete()` method to selectively remove processed images. This is useful for reducing dataset size or cleaning up unwanted patches.

#### Method signature

```python
def delete(self, deletion_factor: int = 2) -> None:
    """
    Delete every "deletion_factor" items from the images in processed folder.
    """
```

#### Parameters

- **`deletion_factor`** (int, default=2): Determines which items to delete
  - `deletion_factor=2`: Deletes every 2nd item (removes ~50% of images)
  - `deletion_factor=3`: Deletes every 3rd item (removes ~33% of images)
  - `deletion_factor=4`: Deletes every 4th item (removes ~25% of images)

#### Examples

```python
from modules.normalization import Normalization

# Create an instance
normalizer = Normalization(folder_name="my_images")

# Preprocess images first
normalizer.preprocess()

# Delete every 2nd image (default, removes ~50%)
normalizer.delete()

# Delete every 3rd image (removes ~33%)
normalizer.delete(deletion_factor=3)

# Delete every 4th image (removes ~25%)
normalizer.delete(deletion_factor=4)
```

#### Exception Handling

The `delete()` method will raise exceptions in the following cases:

- **`FileNotFoundError`**: Raised if the processed folder does not exist. Make sure to call `.preprocess()` first.
  ```python
  normalizer.delete()  # Raises FileNotFoundError if preprocess() wasn't called
  ```

- **`ValueError`**: Raised if the processed folder is empty.
  ```python
  normalizer.delete()  # Raises ValueError if no images are in the processed folder
  ```

#### Important notes

- **The processed folder must exist**: Call `.preprocess()` before using `.delete()`
- **Deletion is permanent**: Deleted images cannot be recovered. Consider backing up your processed folder first.
- **Operates on processed images only**: This method only affects images in the processed folder, not the original raw images.

---

**Module notes:**
- **`normalization.py`**: Contains the `Normalization()` class - this is the main interface you should use
- **`process_fcns.py`**: Internal helper functions used by the `Normalization()` class - not meant to be used directly
- **`deprecated.py`**: Discontinued work - do not use

## Using Jupyter Notebooks

This project supports development and testing with Jupyter notebooks in VS Code.

### Setup

1. **Install ipykernel** (development dependency):
   ```bash
   poetry add ipykernel --group dev
   ```

2. **Restart VS Code** or reload the window:
   - Press `Cmd/Ctrl + Shift + P`
   - Select "Developer: Reload Window"

3. **Select the Poetry kernel**:
   - Open or create a `.ipynb` file
   - Click on the kernel picker in the top-right corner
   - Select the Poetry virtual environment (e.g., `computer-vision-tools-XXXXXXXX-py3.11`)

### Using your modules in notebooks

Once the kernel is set up, you can import and use the `Normalization` class:

```python
# Import the Normalization class
from modules.normalization import Normalization

# Create an instance pointing to a folder in Raw_imgs/
normalizer = Normalization(folder_name="my_images")

# Process the images
normalizer.preprocess()

# The processed 1024x1024 patches will be saved in:
# Images/Processed_imgs/my_images_processed/

# Delete every 2nd image
normalizer.delete(deletion_factor=2)
```

**Note:** Do not directly import or use functions from `process_fcns.py` - these are internal helper functions used by the `Normalization` class.

### Troubleshooting

If VS Code doesn't detect the Poetry environment:

1. Manually register the kernel:
   ```bash
   poetry run python -m ipykernel install --user --name=computer-vision-tools
   ```

2. Restart VS Code and select "computer-vision-tools" from the kernel picker.

## Project Structure

```
Computer-Vision-Tools/
├── Images/
│   ├── Processed_imgs/        # Output: 1024x1024 image patches
│   └── Raw_imgs/               # Input: High-resolution images (e.g., 6000x8000)
├── modules/
│   ├── normalization.py        # Main module: Normalization() class
│   ├── process_fcns.py         # Internal helper functions (do not use directly)
│   ├── deprecated.py           # Discontinued work (do not use)
│   └── ...
├── poetry.lock
├── pyproject.toml
└── README.md
```

### Folder workflow

1. **Raw_imgs/**: Create subfolders here containing your high-resolution images
   - Example: `Raw_imgs/dataset1/`, `Raw_imgs/dataset2/`

2. **Processed_imgs/**: Automatically created when running `.preprocess()`
   - Output structure: `Processed_imgs/dataset1_processed/`, `Processed_imgs/dataset2_processed/`
   - Each contains 1024x1024 patches cut from the original images

## Additional Resources

- [Poetry Basic Usage](https://python-poetry.org/docs/basic-usage/)
- [Poetry Commands](https://python-poetry.org/docs/cli/)
