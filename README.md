System requirement: Ubuntu 20.04/Ubuntu 22.04, Cuda 12.1
While executing make sure you have A100 GPU 
Create conda environment:

  conda create -n hallo python=3.10
  conda activate hallo
Install packages with pip

  pip install -r requirements.txt
  pip install .
Besides, ffmpeg is also needed:

  apt-get install ffmpeg
🗝️️ Usage
The entry point for inference is scripts/inference.py. Before testing your cases, two preparations need to be completed:

Download all required pretrained models.
Prepare source image and driving audio pairs.
Run inference.
📥 Download Pretrained Models
You can easily get all pretrained models required by inference from our HuggingFace repo.

Clone the pretrained models into ${PROJECT_ROOT}/pretrained_models directory by cmd below:

git lfs install
git clone https://huggingface.co/fudan-generative-ai/hallo pretrained_models
Or you can download them separately from their source repo:

hallo: Our checkpoints consist of denoising UNet, face locator, image & audio proj.
audio_separator: Kim_Vocal_2 MDX-Net vocal removal model. (Thanks to KimberleyJensen)
insightface: 2D and 3D Face Analysis placed into pretrained_models/face_analysis/models/. (Thanks to deepinsight)
face landmarker: Face detection & mesh model from mediapipe placed into pretrained_models/face_analysis/models.
motion module: motion module from AnimateDiff. (Thanks to guoyww).
sd-vae-ft-mse: Weights are intended to be used with the diffusers library. (Thanks to stablilityai)
StableDiffusion V1.5: Initialized and fine-tuned from Stable-Diffusion-v1-2. (Thanks to runwayml)
wav2vec: wav audio to vector model from Facebook.
Finally, these pretrained models should be organized as follows:

./pretrained_models/
|-- audio_separator/
|   |-- download_checks.json
|   |-- mdx_model_data.json
|   |-- vr_model_data.json
|   `-- Kim_Vocal_2.onnx
|-- face_analysis/
|   `-- models/
|       |-- face_landmarker_v2_with_blendshapes.task  # face landmarker model from mediapipe
|       |-- 1k3d68.onnx
|       |-- 2d106det.onnx
|       |-- genderage.onnx
|       |-- glintr100.onnx
|       `-- scrfd_10g_bnkps.onnx
|-- motion_module/
|   `-- mm_sd_v15_v2.ckpt
|-- sd-vae-ft-mse/
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
|-- stable-diffusion-v1-5/
|   `-- unet/
|       |-- config.json
|       `-- diffusion_pytorch_model.safetensors
`-- wav2vec/
    `-- wav2vec2-base-960h/
        |-- config.json
        |-- feature_extractor_config.json
        |-- model.safetensors
        |-- preprocessor_config.json
        |-- special_tokens_map.json
        |-- tokenizer_config.json
        `-- vocab.json
🛠️ Prepare Inference Data
Hallo has a few simple requirements for input data:

For the source image:

It should be cropped into squares.
The face should be the main focus, making up 50%-70% of the image.
The face should be facing forward, with a rotation angle of less than 30° (no side profiles).
For the driving audio:

It must be in WAV format.
It must be in English since our training datasets are only in this language.
Ensure the vocals are clear; background music is acceptable.
We have provided some samples for your reference.

🎮 Run Inference
Simply to run the scripts/inference.py and pass source_image and driving_audio as input:

python scripts/inference.py --source_image examples/reference_images/1.jpg --driving_audio examples/driving_audios/1.wav
Animation results will be saved as ${PROJECT_ROOT}/.cache/output.mp4 by default. You can pass --output to specify the output file name. You can find more examples for inference at examples folder.

For more options:

usage: inference.py [-h] [-c CONFIG] [--source_image SOURCE_IMAGE] [--driving_audio DRIVING_AUDIO] [--output OUTPUT] [--pose_weight POSE_WEIGHT]
                    [--face_weight FACE_WEIGHT] [--lip_weight LIP_WEIGHT] [--face_expand_ratio FACE_EXPAND_RATIO]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
  --source_image SOURCE_IMAGE
                        source image
  --driving_audio DRIVING_AUDIO
                        driving audio
  --output OUTPUT       output video file name
  --pose_weight POSE_WEIGHT
                        weight of pose
  --face_weight FACE_WEIGHT
                        weight of face
  --lip_weight LIP_WEIGHT
                        weight of lip
  --face_expand_ratio FACE_EXPAND_RATIO
                        face region
