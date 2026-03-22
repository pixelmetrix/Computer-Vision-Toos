from PIL import Image, ImageOps
from pathlib import Path
from .process_fcns import *

class Normalize(): 
    """
    Class to preprocess (i.e. normalize) images into sub, 1024x1024 images.
    """
    def __init__(self,folder_name: str):
        self.folder_name = str(folder_name) #track folder name
        Image_path = Path(__file__).parents[1] / 'Images' #path to parent image folder
        #route to raw images:
        self.raw_path = Image_path / 'Raw_imgs' / self.folder_name
        #route to processed images:
        self.processed_path = Image_path / 'Processed_imgs' / f'{self.folder_name}_processed'

        # Check if the raw folder exists
        if not self.raw_path.exists():
            raise FileNotFoundError(f"The folder '{self.raw_path}' does not exist.")

    def preprocess(self) -> None:
        """
        cut images, pad them, and place them into processed folder.
        the border value for pad is calculated to be a multiple of 32.
        of the image dimension. this logic is implemented in the mul_32 function
        This works as long as the images to be processed are square.
        """        
        try:

            self.processed_path.mkdir(parents=True, exist_ok=True) # exist_ok allows method to proceed even if directory already exists 

            for index, img_route in enumerate(self.raw_path.iterdir()):
                print(f"Processing image: {img_route}")
                # Instanciate image object
                input_image = Image.open(self.raw_path / img_route.name)
                for index2, imagen in enumerate(divide_rectangular(input_image)):
                    # Find img dimension and, if not multiple of 32, find the next multiple.
                    # Find the difference between the multiple and img dimension. divide by to
                    temp = int((mul_32(imagen.size[0]) - imagen.size[0]) / 2) #increases inference time significantly
                    imagen = ImageOps.expand(imagen, border=temp, fill='gray')  # pad images
                    imagen.save(self.processed_path / f'{self.folder_name}_{index:04d}_{index2:02d}.jpeg')
              
            print('Preprocessing Successful')

        except Exception as e:
            print(f"Overall preprocessing error: {e}")
            raise

    def delete(self, deletion_factor: int = 2) -> None:
        """
        Delete every "deletion_factor" items from the images in processed folder.
        """    
        # Raise exception if processed folder does not exist
        if not self.processed_path.exists():
            raise FileNotFoundError(f"The folder '{self.processed_path}' does not exist. Make sure to preprocess images first")

        # List all items in a directory
        items = list(self.processed_path.iterdir())
        
        # Raise exception if folder is empty
        if not items:
            raise ValueError("The folder is empty")
        
        # Delete every deletion_factor-th item
        for i in range(deletion_factor - 1, len(items), deletion_factor):
            items[i].unlink()

if __name__ == "__main__":

    my_objct = Normalize("test")
    my_objct.delete()