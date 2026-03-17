# Demo Samples
Demo samples can be found on https://voiceprivacy.github.io/any-to-any-speaker-attribute-perturbation
# Start Inference
The model can be downloaded at the link https://huggingface.co/Yonger123/ecapa_adv/tree/main. Put it in root dir. Specify the original sample path and the output anonymized speech path. Enter the following command in the command line:
```
python inference.py \
    --original_dir original_sample.wav \ 
    --adversarial_dir output_sample.wav
```
You can get anonymized speech.