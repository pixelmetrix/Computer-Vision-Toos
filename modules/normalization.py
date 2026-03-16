from PIL import Image, ImageOps
from pathlib import Path
from .process_fcns import *

class Normalize(): 
    """
    Abstract super class whose instances specify the particular raw folder we'll process. 
    Processed images will be pasted in Processed_imgs folder. Subclasses have a slight
    modification for processing images, depending on our use (training or inference).
    Note: this class is not meant to be instanciated
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

    def preprocess(self):
        """
        cut images, pad them, and place them into processed folder.
        the border value for pad is calculated to be a multiple of 32.
        of the image dimension. this logic is implemented in the mul_32 function
        This works as long as the images to be processed are square.
        Note: method changes slightly depending on subclass.
        """        
        try:

            self.processed_path.mkdir(parents=True, exist_ok=True) # exist_ok allows method to proceed even if directory already exists 

            for index, img_route in enumerate(self.raw_path.iterdir()):
                print(f"Processing image: {img_route}")
                # Instanciate image object
                input_image = Image.open(self.raw_path / img_route.name)
                # input_image = self.action(input_image)
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

if __name__ == "__main__":

    my_objct = Normalize("test")
    print(my_objct.folder_name)