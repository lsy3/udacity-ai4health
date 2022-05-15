"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import numpy as np

from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from inference.UNetInferenceAgent import UNetInferenceAgent
from utils.volume_stats import Dice3d, Jaccard3d

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"/home/workspace/data"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "/home/workspace/out"

def run_test(test_data):
    print("Testing...")

    inference_agent = UNetInferenceAgent(parameter_file_path='/home/workspace/out/2022-05-14_1417_Basic_unet/model.pth')

    out_dict = {}
    out_dict["volume_stats"] = []
    dc_list = []
    jc_list = []

    # for every in test set
    for i, x in enumerate(test_data):
        pred_label = inference_agent.single_volume_inference(x["image"])

        # We compute and report Dice and Jaccard similarity coefficients which 
        # assess how close our volumes are to each other

        # DONE: Dice3D and Jaccard3D functions are not implemented. 
        #  Complete the implementation as we discussed
        # in one of the course lessons, you can look up definition of Jaccard index 
        # on Wikipedia. If you completed it
        # correctly (and if you picked your train/val/test split right ;)),
        # your average Jaccard on your test set should be around 0.80

        dc = Dice3d(pred_label, x["seg"])
        jc = Jaccard3d(pred_label, x["seg"])
        dc_list.append(dc)
        jc_list.append(jc)

        # STAND-OUT SUGGESTION: By way of exercise, consider also outputting:
        # * Sensitivity and specificity (and explain semantic meaning in terms of 
        #   under/over segmenting)
        # * Dice-per-slice and render combined slices with lowest and highest DpS
        # * Dice per class (anterior/posterior)

        out_dict["volume_stats"].append({
            "filename": x['filename'],
            "dice": dc,
            "jaccard": jc
        })
        print(f"{x['filename']} Dice {dc:.4f} Jaccard {jc:.4f}. {100*(i+1)/len(test_data):.2f}% complete")

    out_dict["overall"] = {
        "mean_dice": np.mean(dc_list),
        "mean_jaccard": np.mean(jc_list)}

    print("\nTesting complete.")    
    return out_dict
    
def filename_to_index(data, filename_list):
    idx_list = []

    for i in filename_list:
        found = False
        for j, d in enumerate(data):
            if(i == d['filename']):
                found = True
                idx_list.append(j)
                break
        if not found:
            print(f"{i} not found")
            
    return idx_list

if __name__ == "__main__":
    # Get configuration

    # Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()
    c.root_dir = "/home/workspace/data"
    c.test_results_dir = "/home/workspace/out"

    # Load data
    print("Loading data...")

    # DONE: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    # test file list here has the same test set as run_ml_pipeline.py for run 2022-05-14_1417
    test_file_list = ["hippocampus_355.nii.gz", "hippocampus_188.nii.gz", "hippocampus_068.nii.gz", "hippocampus_143.nii.gz", "hippocampus_257.nii.gz", "hippocampus_263.nii.gz", "hippocampus_017.nii.gz", "hippocampus_203.nii.gz", "hippocampus_126.nii.gz", "hippocampus_387.nii.gz", "hippocampus_105.nii.gz", "hippocampus_386.nii.gz", "hippocampus_314.nii.gz", "hippocampus_195.nii.gz", "hippocampus_084.nii.gz", "hippocampus_228.nii.gz", "hippocampus_125.nii.gz", "hippocampus_170.nii.gz", "hippocampus_185.nii.gz", "hippocampus_380.nii.gz", "hippocampus_217.nii.gz", "hippocampus_083.nii.gz", "hippocampus_056.nii.gz", "hippocampus_245.nii.gz", "hippocampus_019.nii.gz", "hippocampus_097.nii.gz", "hippocampus_212.nii.gz", "hippocampus_204.nii.gz", "hippocampus_230.nii.gz", "hippocampus_251.nii.gz", "hippocampus_298.nii.gz", "hippocampus_282.nii.gz", "hippocampus_087.nii.gz", "hippocampus_325.nii.gz", "hippocampus_193.nii.gz", "hippocampus_393.nii.gz", "hippocampus_300.nii.gz", "hippocampus_279.nii.gz", "hippocampus_235.nii.gz", "hippocampus_286.nii.gz", "hippocampus_146.nii.gz", "hippocampus_094.nii.gz", "hippocampus_316.nii.gz", "hippocampus_180.nii.gz", "hippocampus_039.nii.gz", "hippocampus_053.nii.gz", "hippocampus_154.nii.gz", "hippocampus_133.nii.gz", "hippocampus_345.nii.gz", "hippocampus_223.nii.gz", "hippocampus_328.nii.gz", "hippocampus_378.nii.gz"]
    
    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality    
    test_idx = filename_to_index(data, test_file_list)
    results_json = run_test(data[test_idx])

    results_json["config"] = vars(c)

    with open(os.path.join(c.test_results_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

