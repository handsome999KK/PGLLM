## ✨ Highlights

- 🚀 **Training-free**: PGLLM is a **fully test-time optimization** framework for 3D point cloud LLMs, requiring **no additional training or fine-tuning**.
- 🧩 **Two baseline backbones supported**: We provide evaluation pipelines/models based on **PointLLM-7B** and **MiniGPT-3D** for easy reproduction across different 3D-LLM baselines.
- 💻 **Single-GPU friendly**: The released testing pipeline can run on **a single RTX 3090**, making reproduction and deployment practical.

## Pointllm-based
### 1、Install packages (you can follow [PointLLM](https://github.com/InternRobotics/PointLLM) to build the env)
```bash
cd pointllm
conda create -n pointllm python=3.10 -y
conda activate pointllm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
### 2、Data Preparation
you need to follow  [PointLLM](https://github.com/InternRobotics/PointLLM) to download ModelNet40 and Objaverse Dataset and put them to the same folder, as for Shapenetcore, you can from [here](https://drive.google.com/drive/folders/1xEblkFTEIdV1IyIlQLi792-lXfoxVCYO?usp=sharing) to download.

### 3、 Pretrained model download
 you can go to [there](https://huggingface.co/RunsenXu) to download the Pretrained model. In our paper, we use the PointLLM_7B_v1.2 as the Pretrained model.

### 4、 PointLLM inference
    Run the following commands to infer the 3D captions:
```bash
export PYTHONPATH=$PWD
python pointllm/eval/eval_modelnet_cls.py --model_name PATH/TO/YOUR/Pretrainedmodel --prompt_index 0 
--dataset_name ModelNet --batch_size 64
```
Afer the that, you will get two files. (1) 3D captions files: ModelNet40_classification_prompt0.json (2)Features of the test samples: concat_f_values_MN.txt. the results will be saved in {model_name}/evaluation as a dict with the following format:

    {
    "prompt": "",
    "results": [
    {
      "object_id": "",
      "ground_truth": "", 
      "model_output": "",
      "label_name": "" # only for classification on modelnet40
    }
    ]
    }


### 5、 LLM inference
    We provide GPT-4 and DeepSeek-V3 for inference. For 3D recognition task, You can run the following commands:
 ```bash
# For 3D recognition task
cd Point-Graph LLM

# use DeepSeek-V3
python deepseek_cls_MN.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.json

# use GPT-4
python GPT_cls_MN.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.json
```
After that, You will get a LLM inference score file: Point-Graph LLM/GPT__results_cls_MN.json(or DeepSeeK-V3)

For 3D OOD detection task, You can run the following commands:
 ```bash
# For 3D OOD detection task
cd Point-Graph LLM

# use DeepSeek-V3 
python deepseek_OOD_MN.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --dataset_split MN1
--PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.

python deepseek_OOD_MN.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --dataset_split MN2
--PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.json

python deepseek_OOD_MN.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --dataset_split MN3
--PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.json

# use DeepSeek-V3 
python GPT_OOD_MN.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --dataset_split MN1
--PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.

python GPT_OOD_MN.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --dataset_split MN2
--PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.

python GPT_OOD_MN.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --dataset_split MN3
--PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.
```
   After that, You will get a LLM inference score file: Point-Graph LLM/GPT__results_OOD_MNx.json(or DeepSeeK-V3)

### 6、 Final inference

 You can run the following following commands to get the final results for 3D OOD detection or For 3D recognition.
 ```bash
# For 3D recognition task
python ACC.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --LLM_results_path PATH/TO/YOUR/GPT__results_cls_MN.json
--PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.

# For 3D OOD detection task
python AUROC.py  --features_path PATH/TO/YOUR/concat_f_values_MN.txt --dataset_split MN1 
--PointLLM_results_path PATH/TO/YOUR/ModelNet_classification_prompt0.  --LLM_results_path PATH/TO/YOUR/GPT__results_OOD_MN1.json
```

### 7、







