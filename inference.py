import sys
import torchaudio
import torch
import yaml
from ecapa_adv import adversary_generator
import argparse

parser = argparse.ArgumentParser(description='Process some variables.')
parser.add_argument('--original_dir', type=str, default="original_sample.wav")
parser.add_argument('--adversarial_dir', type=str, default="adversarial_sample.wav")
args = parser.parse_args()


original_path = args.original_dir
adversarial_path = args.adversarial_dir

with open("final.params", "rb") as f:
    adv_state = torch.load(f, map_location=torch.device("cpu"))
    
with open("adversary_generation_enc_dec.yaml", 'r') as fin:
    cfg = yaml.load(fin, Loader=yaml.FullLoader)

component_state_dict = {}
for key in adv_state.keys():
    if  key.startswith("decoder"):
        continue
    if "global" in key or "mel" in key:
        continue
    component_state_dict[key.replace("perturbation_generator.","")]=adv_state[key]       

adv_model = adversary_generator(**cfg["Adversary_Generator"])
adv_model.load_state_dict(component_state_dict,strict=True)
adv_model.eval()

ori_speech, _ = torchaudio.load(original_path)
adv_wav,_ = adv_model(ori_speech)[0]

torchaudio.save(adversarial_path, adv_wav.detach(), 16000, bits_per_sample = 16)
