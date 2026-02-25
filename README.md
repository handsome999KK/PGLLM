## ✨ Highlights

- 🚀 **Training-free**: PGLLM is a **fully test-time optimization** framework for 3D point cloud LLMs, requiring **no additional training or fine-tuning**.
- 🧩 **Two baseline backbones supported**: We provide evaluation pipelines/models based on **PointLLM-7B** and **MiniGPT-3D** for easy reproduction across different 3D-LLM baselines.
- 💻 **Single-GPU friendly**: The released testing pipeline can run on **a single RTX 3090**, making reproduction and deployment practical.

## Pointllm-based
1、Install packages (you can follow [PointLLM](https://github.com/InternRobotics/PointLLM) to build the env)
```bash
cd pointllm
conda create -n pointllm python=3.10 -y
conda activate pointllm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
2、Data Preparation
you need to follow  [PointLLM](https://github.com/InternRobotics/PointLLM) to download ModelNet40 and Objaverse Dataset and put them to the same folder, as for Shapenetcore, you can from [here](https://drive.google.com/drive/folders/1xEblkFTEIdV1IyIlQLi792-lXfoxVCYO?usp=sharing) to download.

3、 Pretrained model download
 you can go to [there](https://huggingface.co/RunsenXu) to download the Pretrained model. In our paper, we use the PointLLM_7B_v1.2 as the Pretrained model.

 4、 PointLLM inference
    Run the following commands to infer the 3D captions:
```bash
    cd Point-Graph LLM
    export PYTHONPATH=$PWD
    python pointllm/eval/eval_modelnet_cls.py --model_name PATH/TO/YOUR/Pretrainedmodel --prompt_index 0 
    --dataset_name ModelNet --batch_size 64
```
