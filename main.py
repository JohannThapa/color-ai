from color_utils import ColorUtils
from tensorflow_model import TensorFlowModel
from pytorch_model import PyTorchModel

def main():
    # Initialize color utility
    color_utility = ColorUtils(data_path='dta/colors.json')

    # User input
    input_color = input("Enter a color name, hex code, or RGB values (comma-separated): ")

    # Get closest color
    closest_color = color_utility.get_closest_color(input_color)

    # Print closest color details
    print("Closest color found:")
    print(closest_color)

    # Perform color type prediction using TensorFlow model
    tensorflow_model = TensorFlowModel(input_shape=(3,), num_classes=3)
    tensorflow_prediction = tensorflow_model.predict([[closest_color['rgb']]])[0]

    # Perform color type prediction using PyTorch model
    pytorch_model = PyTorchModel(input_size=3, num_classes=3)
    pytorch_prediction = pytorch_model.predict(torch.tensor([closest_color['rgb']]))

    print("Predicted color type (TensorFlow):", tensorflow_prediction)
    print("Predicted color type (PyTorch):", pytorch_prediction.item())

if __name__ == "__main__":
    main()
