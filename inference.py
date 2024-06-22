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
# os.environ['UNI_CKPT_PATH'] = "./resources/uni.bin"
from extract_feature_utils import create_patches_fp
from extract_feature_utils import extract_features_fp
import inference_utils
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

config = "./config/Patch_GCN_v2_test.yaml"
# config = "./config/Patch_GCN_v2.yaml"
# ckpt_path = "./resources/cTranspath_TransMIL_epoch_6_index_0.7628541448058762.pth"
ckpt_path = "./resources/cTranspath_Patch_GCN_v2_test_epoch_1_index_0.6610850636302746.pth"
# ckpt_path = "./resources/cTranspath_Patch_GCN_v2_test_epoch_2_index_0.6731413261888815.pth" # new ckpt
feature_dir = "/tmp/features/pt_files/"


def run():
    # Read the input
    wsi_dir = os.path.join(INPUT_PATH, "images/prostatectomy-wsi")
    wsi_list = sorted(os.listdir(wsi_dir))

    tissue_mask_dir = os.path.join(INPUT_PATH, "images/prostatectomy-tissue-mask")
    tissue_mask_list = sorted(os.listdir(tissue_mask_dir))

    create_patches_fp.create_patches(source=wsi_dir, save_dir='/tmp/features', seg=True, patch=True, patch_size=512, step_size=512) # , patch_size=512, step_size=512)

    save_pt = 'Patch_GCN' not in config
    print('save_pt: ', save_pt)
    if 'cTranspath' in ckpt_path:
        print(f'Extracting features using cTranspath model')
        extract_features_fp.extract_features(data_h5_dir='/tmp/features', data_slide_dir=wsi_dir, slide_ext='.tif', csv_path='/tmp/features/process_list_autogen.csv', feat_dir='/tmp/features', model_name='ctranspath', batch_size=480, target_patch_size=224, save_pt=save_pt) # model_name = 'resnet50_trunc'
    elif 'uni' in ckpt_path:
        print(f'Extracting features using uni model')
        extract_features_fp.extract_features(data_h5_dir='/tmp/features', data_slide_dir=wsi_dir, slide_ext='.tif', csv_path='/tmp/features/process_list_autogen.csv', feat_dir='/tmp/features', model_name='uni_v1', batch_size=512, target_patch_size=224, save_pt=save_pt) # model_name = 'resnet50_trunc'
    else:
        print(f'Extracting features using resnet50 model')
        extract_features_fp.extract_features(data_h5_dir='/tmp/features', data_slide_dir=wsi_dir, slide_ext='.tif', csv_path='/tmp/features/process_list_autogen.csv', feat_dir='/tmp/features', model_name='resnet50_trunc', batch_size=512, target_patch_size=224, save_pt=save_pt) # model_name = 'resnet50_trunc'

    if 'Patch_GCN' in config:
        from extract_feature_utils import h5toPyG

        h5toPyG.createDir_h5toPyG(h5_path='/tmp/features/h5_files', save_path='/tmp/features/pt_files')
        
    
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



if __name__ == "__main__":
    if os.path.exists('/tmp/features'):
        os.system('rm -rf /tmp/features')
    print(os.listdir('/tmp'))
    run()
    # raise SystemExit(run())
