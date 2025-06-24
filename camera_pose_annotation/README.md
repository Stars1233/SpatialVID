# Step2_megasam
<!-- This code is forked from git@github.com:mega-sam/mega-sam.git, only for academic purpose.-- [wjh] -->

<!-- # 🚧 This repository is still not done and being uploaded, please stand by. 🚧  -->

[Project Page](https://mega-sam.github.io/index.html) | [Paper](https://arxiv.org/abs/2412.04463)

Clone the repository:
```bash
git clone https://github.com/FelixYuan-YF/step2_megasam.git
cd step2_megasam
```

## Instructions for installing dependencies
### Python Environment

1.  Create conda environment similar to the [previous step](https://github.com/FelixYuan-YF/step1_scoring): 
    ```bash
    conda create -n megasam python=3.10.13
    conda activate megasam
    pip install -r requirements.txt
    pip install -r requirements_megasam.txt
    ```

2.  Compile the extensions for the camera tracking module:
    ```bash
    cd base
    python setup.py install
    ```

3. viser(not necessary if you only want to run the model):
   ```bash
   pip install -e viser
   ```

### Downloading pretrained checkpoints

You can download the pretrained checkpoints by running the following script:
```bash
bash tools/checkpoints_download.sh
```

Or you can download the checkpoints manually from the following links and place them in the `checkpoints` directory:

+ [Depth Anything V2](https://depth-anything-v2.github.io/)

    | Model | Params | Checkpoint |
    |:-|-:|:-:|
    | Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
    | Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
    | Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |

+ [UniDepthV2](https://lpiccinelli-eth.github.io/pub/unidepth/)

    <table border="0">
        <tr>
            <th>Model</th>
            <th>Backbone</th>
            <th>Name</th>
        </tr>
        <tr>
            <td rowspan="3"><b>UnidepthV2</b></td>
            <td>ViT-S</td>
            <td><a href="https://huggingface.co/lpiccinelli/unidepth-v2-vits14">unidepth-v2-vits14</a></td>
        </tr>
        <tr>
            <td>ViT-B</td>
            <td><a href="https://huggingface.co/lpiccinelli/unidepth-v2-vitb14">unidepth-v2-vits14</a></td>
        </tr>
        <tr>
            <td>ViT-L</td>
            <td><a href="https://huggingface.co/lpiccinelli/unidepth-v2-vitl14">unidepth-v2-vitl14</a></td>
        </tr>
    </table>

+ [Raft](https://github.com/princeton-vl/RAFT?tab=readme-ov-file)

    Download the `raft-thing.pth` from [RAFT checkpoint](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)

### Running
Before officially running the model, you can run the following command to test if the model is working properly:
```bash
bash run_test.sh
```
or
```bash
python run_inference.py test/csv/videos.csv --dir_path test/outputs --checkpoints_path checkpoints --gpu_id 0 --num_workers 1 --extract_interval 0.2 --dpt_encoder vitl --all
```

After the test is successful, you can run the following command to run the model:
```bash
python run_inference.py path/to/meta.csv --dir_path outputs --checkpoints_path checkpoints --gpu_id 0,1,2,3,4,5,6,7 --num_workers 16 --extract_interval 0.2 --dpt_encoder vitl
```

### Visualizing
Run the following command to visualize the output
```bash
python viser_batch.py
```

