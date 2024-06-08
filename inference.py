"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy
import os
from extract_feature_utils import create_patches_fp
from extract_feature_utils import extract_features_fp
import inference_utils
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

def run():
    # Read the input
    # input_prostatectomy_tissue_whole_slide_image = load_image_file_as_array(
    #     location=INPUT_PATH / "images/prostatectomy-wsi",
    # )
    # input_prostatectomy_tissue_mask = load_image_file_as_array(
    #     location=INPUT_PATH / "images/prostatectomy-tissue-mask",
    # )
    wsi_dir = os.path.join(INPUT_PATH, "images/prostatectomy-wsi")
    wsi_list = sorted(os.listdir(wsi_dir))

    tissue_mask_dir = os.path.join(INPUT_PATH, "images/prostatectomy-tissue-mask")
    tissue_mask_list = sorted(os.listdir(tissue_mask_dir))

    print(f'wsi_list: {wsi_list}; tissue_mask_list: {tissue_mask_list}')
    
    # Process the inputs: any way you'd like
    # _show_torch_cuda_info()

    create_patches_fp.create_patches(source=wsi_dir, save_dir='/tmp/features', seg=True, patch=True) # , patch_size=512, step_size=512)

    extract_features_fp.extract_features(data_h5_dir='/tmp/features', data_slide_dir=wsi_dir, slide_ext='.tif', csv_path='/tmp/features/process_list_autogen.csv', feat_dir='/tmp/features', model_name='resnet50_trunc', batch_size=512, target_patch_size=224)

    print(f'extracted features: {os.listdir("/tmp/features")}')

    # with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
    #     print(f.read())

    config = "./config/DSMIL.yaml"
    ckpt_path = "./resources/epoch_21_index_0.7053072625698324.pth"
    feature_dir = "/tmp/features/pt_files/"
    
    result_dict_list = inference_utils.inference(config, feature_dir, ckpt_path)

    assert len(result_dict_list) == 1
    
    for result_dict in result_dict_list:
        # For now, let us set make bogus predictions
        output_overall_survival_years = result_dict['predicted_time']

        # Save your output
        write_json_file(
            location=OUTPUT_PATH / "overall-survival-years.json",
            content=output_overall_survival_years
        )
    
    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    run()
    # raise SystemExit(run())
