import os
import gradio as gr
import torch
import torchaudio
import csv
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from utils.prompter import Prompter
import datetime
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = "/fs/nexus-projects/brain_project/acl_sk_24/GAMA//train_script/Llama-2-7b-chat-hf-qformer/"

prompter = Prompter('alpaca_short')
tokenizer = LlamaTokenizer.from_pretrained(base_model)

model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float32)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
temp, top_p, top_k = 0.1, 0.95, 500

eval_mdl_path = '/fs/gamma-projects/audio/gama/new_data/stage4_all_mix_new/checkpoint-46800/pytorch_model.bin'
state_dict = torch.load(eval_mdl_path, map_location='cpu')
msg = model.load_state_dict(state_dict, strict=False)

model.is_parallelizable = True
model.model_parallel = True
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()
cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_save_path = './inference_log/'
if not os.path.exists(log_save_path):
    os.mkdir(log_save_path)
csv_save_path = os.path.join(log_save_path, f"inference_results_{cur_time}.csv")

SAMPLE_RATE = 16000
AUDIO_LEN = 1.0

def load_audio(filename):
    waveform, sr = torchaudio.load(filename)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                              use_energy=False, window_type='hanning',
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    fbank = (fbank + 5.081) / 4.4849
    return fbank

def predict_multiple(audio_paths, question):
    with open(csv_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Audio File", "Precition"])
        
        for audio_path in audio_paths:
            audio_info, output = predict(audio_path, question)
            writer.writerow([audio_path, output])

    print(f"Results saved to {csv_save_path}")
    return csv_save_path

def predict(audio_path, question):
    instruction = question
    prompt = prompter.generate_prompt(instruction, None)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    if audio_path != 'empty':
        cur_audio_input = load_audio(audio_path).unsqueeze(0).to(device)
    else:
        cur_audio_input = None

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=400,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids.to(device),
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=400,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)[len(prompt)+6:-4] # trim <s> and </s>
    return audio_path, output

# GUI for multiple files
demo = gr.Interface(fn=predict_multiple,
                    inputs=[gr.File(file_count="multiple", type="./audio-test/mutox-dataset/toxic"), gr.Textbox(default="Is the audio toxic?")],
                    outputs=[gr.File(label="CSV Results")],
                    cache_examples=True,
                    title="GAMA Batch Prediction",
                    description="Input multiple audio files for batch toxicity analysis. Results will be saved to a CSV file.")
demo.launch(debug=True, share=True)
