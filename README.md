## ✨ Highlights

- 🚀 **Training-free**: PGLLM is a **fully test-time optimization** framework for 3D point cloud LLMs, requiring **no additional training or fine-tuning**.
- 🧩 **Two baseline backbones supported**: We provide evaluation pipelines/models based on **PointLLM-7B** and **MiniGPT-3D** for easy reproduction across different 3D-LLM baselines.
- 💻 **Single-GPU friendly**: The released testing pipeline can run on **a single RTX 3090**, making reproduction and deployment practical.

## Pointllm-based
Install packages (you can follow PointLLM to build the env, link:https://github.com/InternRobotics/PointLLM)
```bash
cd pointllm
conda create -n pointllm python=3.10 -y
conda activate pointllm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
