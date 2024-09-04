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
# import SimpleITK
import numpy
import os
import torch
# os.environ['UNI_CKPT_PATH'] = "./resources/uni.bin"
from extract_feature_utils import create_patches_fp
from extract_feature_utils import extract_features_fp
import inference_utils
INPUT_PATH = Path("/input")
# INPUT_PATH = Path("./test/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

# config_list = ["./config/TransMIL.yaml", "./config/Patch_GCN_v2_test.yaml", "./config/TransMIL_1024.yaml"]
# config = "./config/Patch_GCN_v2.yaml"
# ckpt_path = "./resources/cTranspath_TransMIL_epoch_6_index_0.7628541448058762.pth"
# ckpt_path_list = ["./resources/cTranspath_TransMIL_epoch_6_index_0.7628541448058762.pth", "./resources/cTranspath_Patch_GCN_v2_test_epoch_1_index_0.6610850636302746.pth", "resources/resnet50_TransMIL_epoch_15_index_0.7373134328358208.pth"]
# ckpt_path = "./resources/cTranspath_Patch_GCN_v2_test_epoch_2_index_0.6731413261888815.pth" # new ckpt
# 0716
# config_list = ["./config/ABMIL.yaml", "./config/TransMIL_cat.yaml", "./config/TransMIL.yaml", "config/TransMIL_multi_scale.yaml"]
# ckpt_path_list = ["./resources/ABMIL_cTranspath_1024_epoch_25_index_0.7077.pth", "./resources/TransMIL_cTranspath_512_2048_epoch_25_index_0.7604.pth", "./resources/TransMIL_cTranspath_1024_epoch_5_index_0.8031.pth", "./resources/TransMIL_fusion_cTranspath_512_2048_epoch_21_index_0.7692.pth"]
# config_list = [
#     "./config/ABMIL.yaml", 
#     "./config/DeepGraphConv2.yaml", 
#     "./config/Patch_GCN_v1.yaml", 
#     "./config/Patch_GCN_v1.yaml", 
#     "./config/TransMIL_fusion_scale.yaml", 
#     "./config/TransMIL_fusion_scale.yaml"
# ]
# ckpt_path_list = [
#     "./resources/ABMIL_cTranspath_1024_epoch_24_index_0.7150.pth", 
#     "./resources/DeepGraphConv_cTranspath_1024_epoch_18_index_0.6965.pth", 
#     "./resources/Patch_GCN_cTranspath_1024_epoch_15_index_0.6405.pth", 
#     "./resources/Patch_GCN_cTranspath_1024_epoch_25_index_0.7480.pth", 
#     "./resources/TransMIL_cTranspath_512_2048_epoch_22_index_0.8038.pth", 
#     "./resources/TransMIL_cTranspath_512_2048_epoch_1_index_0.7065.pth"
# ]
config_list = [
    "./config/ABMIL.yaml", 
]
ckpt_path_list = [
    "./resources/ABMIL_cTranspath_1024_epoch_27_index_0.7275.pth", 
]
process_step_size = [] # 20x, 10x

assert len(config_list) == len(ckpt_path_list)

def run():
    # Read the input
    wsi_dir = os.path.join(INPUT_PATH, "images/prostatectomy-wsi")
    wsi_list = sorted(os.listdir(wsi_dir))

    for config, ckpt_path in zip(config_list, ckpt_path_list):
        print(config, ckpt_path)
        if 'cTranspath' in ckpt_path:
            model_name = 'cTranspath'
        elif 'uni' in ckpt_path:
            model_name = 'uni'
        else:
            model_name = 'resnet50'
        step_size_str = ckpt_path.split(model_name + '_')[-1].split('_epoch')[0]
        step_size_list = [int(i) for i in step_size_str.split('_')]

        # extract feature
        for step_size in step_size_list:
            if step_size not in process_step_size:
                process_step_size.append(step_size)
                coord_save_dir = f"/tmp/cords_{step_size}"
                create_patches_fp.create_patches(source=wsi_dir, save_dir=coord_save_dir, seg=True, patch=True, patch_size=step_size, step_size=step_size)
        
                feat_dir = f'/tmp/{model_name}_features_{step_size}'
                if not os.path.exists(feat_dir):
                    if model_name == 'cTranspath':
                        print(f'Extracting features using cTranspath model')
                        extract_features_fp.extract_features(data_h5_dir=coord_save_dir, data_slide_dir=wsi_dir, slide_ext='.tif', csv_path=os.path.join(coord_save_dir, 'process_list_autogen.csv'), feat_dir=feat_dir, model_name='ctranspath', batch_size=495, target_patch_size=224, save_pt=True)
                    elif model_name == 'uni':
                        print(f'Extracting features using uni model')
                        extract_features_fp.extract_features(data_h5_dir=wsi_dir, data_slide_dir=wsi_dir, slide_ext='.tif', csv_path=os.path.join(coord_save_dir, 'process_list_autogen.csv'), feat_dir=feat_dir, model_name='uni_v1', batch_size=512, target_patch_size=224, save_pt=True)
                    else:
                        print(f'Extracting features using resnet50 model')
                        extract_features_fp.extract_features(data_h5_dir=coord_save_dir, data_slide_dir=wsi_dir, slide_ext='.tif', csv_path=os.path.join(coord_save_dir, 'process_list_autogen.csv'), feat_dir=feat_dir, model_name='resnet50_trunc', batch_size=512, target_patch_size=224, save_pt=True)

        if 'Patch_GCN' in config or 'DeepGraphConv' in config:
            try:
                h5toPyG
            except:
                from extract_feature_utils import h5toPyG
            for step_size in step_size_list:
                feat_dir = f'/tmp/{model_name}_features_{step_size}'
                h5_path = os.path.join(feat_dir, 'h5_files')
                save_path = os.path.join(feat_dir, 'graph_pt_files')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    h5toPyG.createDir_h5toPyG(h5_path=h5_path, save_path=save_path)

    result_list = []
    for config, ckpt_path in zip(config_list, ckpt_path_list): # for each MIL/pretrained_model
        if 'cTranspath' in ckpt_path:
            model_name = 'cTranspath'
        elif 'uni' in ckpt_path:
            model_name = 'uni'
        else:
            model_name = 'resnet50'

        patch_size_str = ckpt_path.split(model_name + '_')[-1].split('_epoch')[0]
        patch_size_list = [int(i) for i in patch_size_str.split('_')]
        
        if 'Patch_GCN' in ckpt_path or 'DeepGraphConv' in ckpt_path:
            graph_dir_list = [f'/tmp/{model_name}_features_{patch_size}/graph_pt_files' for patch_size in patch_size_list]
            result = inference_utils.inference(config, graph_dir_list, ckpt_path)[0]['predicted_time']
        else:
            feat_dir_list = [f'/tmp/{model_name}_features_{patch_size}/pt_files' for patch_size in patch_size_list]
            # print(f'ckpt: {ckpt_path}, feat_dir_list: {feat_dir_list}')
            result = inference_utils.inference(config, feat_dir_list, ckpt_path)[0]['predicted_time']
        torch.cuda.empty_cache()
        result_list.append(result)

    # For now, let us set make bogus predictions
    output_overall_survival_years = sum(result_list) / len(result_list)

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


# def load_image_file_as_array(*, location):
#     # Use SimpleITK to read a file
#     input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
#     result = SimpleITK.ReadImage(input_files[0])

#     # Convert it to a Numpy array
#     return SimpleITK.GetArrayFromImage(result)



if __name__ == "__main__":
    # os.system('rm -rf /tmp/*')
    os.system('nvidia-smi')
    raise SystemExit(run())
