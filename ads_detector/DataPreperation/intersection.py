import json
import os
import shutil

json_folder_path = "../Outputs"
cropped_folder_path = "../Cropped"

# List all the JSON files in the folder
json_files = [filename for filename in os.listdir(json_folder_path) if filename.endswith(".json")]

output_folder_path = "../final_jsons"

if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

for json_file in json_files:
    #Stripped JSON_file name, and stripped cropped_image_name are same other than an extra _cropped in the image, copy these files to another folder
    image_file_name = os.path.splitext(json_file)[0] + "_cropped.png"
    image_file_path = os.path.join(cropped_folder_path, image_file_name)

    if os.path.exists(image_file_path):
        # Copy the JSON file to the output folder
        json_file_path = os.path.join(json_folder_path, json_file)
        output_json_file_path = os.path.join(output_folder_path, json_file)

        # Use shutil.copy to copy the file
        shutil.copy(json_file_path, output_json_file_path)

        print(f"JSON file copied: {json_file}")
    else:
        print(f"Corresponding cropped image not found for: {json_file}")

print("Copying completed.")