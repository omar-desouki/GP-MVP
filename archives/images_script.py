import os
import cv2
import subprocess


def resize_and_save_images(input_dir, output_dir, size=(62, 90)):
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} does not exist.")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, size)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, resized_img)
                print(f"Saved resized image to {output_path}")


print("os.getcwd():", os.getcwd())

# input_directory = "./input_imgs"
# output_directory = "./ESRGAN/LR"
input_directory = "../input_imgs"
output_directory = "./LR"
# test_directory = "./ESRGAN/test.py"

resize_and_save_images(input_directory, output_directory)

# Downgrade NumPy to a version below 2.0
# subprocess.run(["pip", "install", "numpy<2", "--break-system-packages"])

# # Define the full path to the Python interpreter
# python_interpreter = "/usr/local/bin/python3.12"
# # Define the full path to the test.py script
# test_script_path = "./MVP/ESRGAN/test.py"
# # Run the test.py script using the full path to the Python interpreter
# subprocess.run([python_interpreter, test_script_path])

# # Run the test.py script in the esrgan folder using the full path to the Python interpreter
# python_interpreter = "/usr/local/bin/python3.12"
# subprocess.run([python_interpreter, test_directory])
