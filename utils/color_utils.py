import json
import numpy as np

class ColorUtils:
    def __init__(self, data_path):
        """
        Initializes with color data from the JSON.
        Args:
        - data_path (str): Path to the JSON file containing color data.
        """
        with open(data_path, 'r') as file:
            self.colors_data = json.load(file)

    def hex_to_rgb(self, hex_code):
        """
        Converts a hexadecimal color code to RGB format.
        Args:
        - hex_code (str): Hexadecimal color code.
        Returns:
        - tuple: RGB color values.
        """
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

    def get_closest_color(self, input_color):
        """
        Finds the closest color in the dataset based on input color.
        Args:
        - input_color (str): Input color (name, hex code, or RGB values).
        Returns:
        - dict: Closest color data from the dataset.
        """
        # Convert input color to RGB format
        if input_color.startswith('#'):
            input_rgb = self.hex_to_rgb(input_color)
        elif input_color.lower() in self.colors_data:
            input_rgb = self.hex_to_rgb(self.colors_data[input_color.lower()]['hex'])
        else:
            input_rgb = tuple(map(int, input_color.split(',')))

        input_rgb = np.array(input_rgb)
        
        # Find the closest color in the dataset based on Euclidean distance
        closest_color = None
        min_distance = float('inf')
        for color in self.colors_data.values():
            color_rgb = np.array(color['rgb'])
            distance = np.linalg.norm(input_rgb - color_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        return closest_color
