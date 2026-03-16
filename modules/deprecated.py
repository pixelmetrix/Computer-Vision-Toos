from PIL import Image, ImageOps
from pathlib import Path
from .process_fcns import *
from collections import Counter, defaultdict
import requests
from ..model_api import predict_local
import pillow_heif
from abc import abstractmethod, ABC
import time
pillow_heif.register_heif_opener()

class Super_process(ABC): 
    """
    Abstract super class whose instances specify the particular raw folder we'll process. 
    Processed images will be pasted in Processed_imgs folder. Subclasses have a slight
    modification for processing images, depending on our use (training or inference).
    Note: this class is not meant to be instanciated
    """
    def __init__(self,folder_name: str):
        self.folder_name = str(folder_name) #track folder name
        Image_path = Path(__file__).parents[2] / 'Images' #path to parent image folder
        #route to raw images:
        self.raw_path = Image_path / 'Raw_imgs' / self.folder_name
        #route to processed images:
        self.processed_path = Image_path / 'Processed_imgs' / f'{self.folder_name}_processed'

        # Check if the raw folder exists
        if not self.raw_path.exists():
            raise FileNotFoundError(f"The folder '{self.raw_path}' does not exist.")

    @abstractmethod
    def action(self,input_image):
        """
        No need for error hanlding. Metaclass definition will raise exception if 
        we try to instantiate Super_process()
        """
        pass

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
                input_image = self.action(input_image)
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

class Training(Super_process):
    """
    Subclass with variation on process images: no prepadding
    Less stable, more unpredictable class for production inferences.
    """
    def action(self,input_image):
        return input_image
class Inference(Super_process, ABC):
    """
    Subclass variation on process_images method: pad image to make square.
    """
    def action(self,input_image): 
        #increases inference time by 2x
        padding = inference_pad(input_image) #complete image making it square. 
        return ImageOps.expand(input_image,border = padding, fill= 'gray')
        # return input_image
    
    @abstractmethod
    def action2(self,image_data):
        """
        Class instantiation will raise an error because the interpreter will read abstract method.
        """
        pass

    def predict(self):  
        # Check if the raw folder exists
        if not self.processed_path.exists():
            raise FileNotFoundError(f"The folder '{self.folder_name}_proc' does not exist.")
        
        print('4. Predicting ...')
        object_counts = Counter()
        
        for img in sorted(self.processed_path.iterdir()):

            #look for position of original image in absolute path
            current_pos =len(str(self.processed_path / f'{self.folder_name}_'))
            str_image = str(img)
            original_image_id = str_image[current_pos:current_pos+4] #retrieve original image identifier
            sub_image_count = self.action2(str_image)       
            object_counts[original_image_id] += int(sub_image_count)
            print("OBJECT COUNT: ", object_counts)

        print(f'Inference completed on {self.folder_name}_proc')

        return object_counts
    
    def batch_predict(self):  
        # Check if the raw folder exists
        if not self.processed_path.exists():
            raise FileNotFoundError(f"The folder '{self.folder_name}_proc' does not exist.")
        
        print('3. Predicting ...')

        # object_counts will map image_id to a Counter of detections
        object_counts = defaultdict(Counter)

        image_bank = sorted(self.processed_path.iterdir())
        image_list = []
        
        for index,img in enumerate(image_bank[:-1]): #only iterate until the value before the last value

            image_id = str(img)[-12:-8]
            next_img = str(image_bank[index+1])
            next_image_id = next_img[-12:-8]

            image_list.append(img)

            if image_id != next_image_id:

                original_image_count = self.action2(image_list)
                object_counts[image_id] += Counter(original_image_count)
                print(f"OBJECT COUNT FOR {image_id}: {object_counts[image_id]}")
                image_list = []

            if index + 1 == len(image_bank) - 1:
                '''This code only executes in the last image'''
                image_list.append(next_img)
                original_image_count = self.action2(image_list)
                object_counts[next_image_id] += Counter(original_image_count)
                print(f"OBJECT COUNT FOR {next_image_id}: {object_counts[next_image_id]}")

        print(f'Inference completed on {self.folder_name}_proc')
        return object_counts
    
class LocalInference(Inference):
    def __init__(self, folder_name, server = 'http://127.0.0.1', port='8080',path = '/batch_predict'):
        super().__init__(folder_name)
        self.server = server
        self.port = port
        self.path = path
        self.url = f"{self.server}:{port}{path}" if port else f"{server}{path}"

    def action2(self,image_data):
        """
        Raises exception if process_images is called from abstract super class instance. 
        """
        packed_json = predict_local(image_data)
        try:
            initial = time.time()
            response = requests.post(self.url,json=packed_json)
            final = time.time()
            print(f"Query prediction backend took: {final - initial:.2f} seconds.")
            return response.json()['predictions'][0]
        
        except Exception as error:
            print('-----> ERROR IN REQUEST TO PREDICTION SERVER:', error)

class CloudRunInference(LocalInference):
    def __init__(self, folder_name, server="http://0.0.0.0:8080", path = '/batch_predict'):
        super().__init__(folder_name, server=server, port=None, path=path)

if __name__ == "__main__":

    import time
    # import cProfile, pstats
    # test_object = Super_process("test") # will raise error because abstract super class cannot be instantiated.
    # test_object = Inference("test") # will raise error because abstract super class cannot be instantiated.
    # test_object = Training("test")
    # profiler = cProfile.Profile()
    # profiler.enable()
    # test_object = CloudRunInference(folder_name='ds_julio4',
    #                                 server="https://prediction-backend-fast-496175836463.us-central1.run.app")
    
    test_object = LocalInference('M18_jul4')
    
    # print(test_object.raw_path)
    start_time0 = time.time()
    test_object.preprocess()
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumtime").print_stats(15)
    end_time0 = time.time()
    elapsed_time0 = end_time0 - start_time0
    print(f"Preprocessing took: {elapsed_time0:.2f} seconds.")
    # Record the start time
    start_time = time.time()
    counts = test_object.batch_predict()
    # Record the end time
    end_time = time.time()
    print(counts)
    elapsed_time = end_time - start_time
    print(f"Query prediction backend took: {elapsed_time:.2f} seconds.")
    