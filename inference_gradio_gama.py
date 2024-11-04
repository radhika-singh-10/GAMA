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
import tempfile
from pydub import AudioSegment

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = "/home/rsingh57/models/gama_checkpoints/Llama-2-7b-chat-hf-qformer/" # "/fs/nexus-projects/brain_project/acl_sk_24/GAMA//train_script/Llama-2-7b-chat-hf-qformer/"

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

eval_mdl_path = '/home/rsingh57/models/gama_checkpoints/checkpoint-2500/pytorch_model.bin' 
# '/fs/gamma-projects/audio/gama/new_data/stage4_all_mix_new/checkpoint-46800/pytorch_model.bin'
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
csv_save_path = os.path.join(log_save_path, f"inference_results_mutox_results_non_toxic.csv")

SAMPLE_RATE = 16000
AUDIO_LEN = 1.0

# def load_audio(filename):
#     waveform, sr = torchaudio.load(filename)
#     if sr != 16000:
#         waveform = torchaudio.functional.resample(waveform, sr, 16000)
#     waveform = waveform - waveform.mean()
#     fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
#                                               use_energy=False, window_type='hanning',
#                                               num_mel_bins=128, dither=0.0, frame_shift=10)
#     target_length = 1024
#     n_frames = fbank.shape[0]
#     p = target_length - n_frames
#     if p > 0:
#         fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
#     elif p < 0:
#         fbank = fbank[0:target_length, :]
#     fbank = (fbank + 5.081) / 4.4849
#     return fbank

def load_audio(filename):
    if filename.endswith('.mp3'):
        audio = AudioSegment.from_mp3(filename)
        temp_wav = tempfile.mktemp(suffix='.wav')
        audio.export(temp_wav, format='wav')
        waveform, sr = torchaudio.load(temp_wav)
        os.remove(temp_wav)
    else:
        waveform, sr = torchaudio.load(filename)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000,
                                              use_energy=False, window_type='hanning',
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p % 2 != 0:
        p += 1  
    if p > 0:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
    elif p < 0:
        fbank = fbank[:target_length, :]
    fbank = (fbank + 5.081) / 4.4849
    return fbank

def predict_multiple(audio_paths, question):
    print(csv_save_path)
    with open(csv_save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Audio File", "Precition"])
        
        for audio_path in audio_paths:
            print(audio_path)
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

def main():
    # Example audio paths and question for prediction
    audio_directory = "/home/rsingh57/audio-test/mutox-dataset/non_toxic"
    audio_paths = [os.path.join(audio_directory, f) for f in os.listdir(audio_directory) if f.endswith('.mp3')]
    question = "Is the audio toxic? If yes, what kind of toxic class does this audio belong to?"

    # Call the predict_multiple function to process audio files and save results to CSV
    predict_multiple(audio_paths, question)

if __name__ == "__main__":
    main()

# GUI for multiple files
# demo = gr.Interface(fn=predict_multiple,
#                     inputs=[
#                         gr.File(file_count="multiple", type="/home/rsingh57/audio-test/mutox-dataset/toxic/"), 
#                             gr.Textbox(default="Is the audio toxic? If yes, what kind of toxic class does this audio belong to?")],
#                     outputs=[gr.File(label="CSV Results")],
#                     cache_examples=True,
#                     title="GAMA Batch Prediction",
#                     description="Input multiple audio files for batch toxicity analysis. Results will be saved to a CSV file.")
# demo = gr.Interface(
#     fn=predict_multiple,
#     inputs=[
#         gr.File(file_count="multiple", type="file"),
#         gr.Textbox(default="Is the audio toxic? If yes, what kind of toxic class does this audio belong to?")
#     ],
#     outputs=[gr.File(label="CSV Results")],
#     cache_examples=True,
#     title="GAMA Batch Prediction",
#     description="Input multiple audio files for batch toxicity analysis. Results will be saved to a CSV file."
# )
# link = "https://github.com/Sreyan88/GAMA"
# text = "[Github]"
# paper_link = "https://sreyan88.github.io/gamaaudio/"
# paper_text = "[Paper]"
# demo = gr.Interface(fn=predict,
#                     inputs=[gr.Audio(type="filepath"), gr.Textbox(value='Describe the audio.', label='Edit the textbox to ask your own questions!')],
#                     outputs=[gr.Textbox(label="Audio Meta Information"), gr.Textbox(label="GAMA Output")],
#                     cache_examples=True,
#                     title="Quick Demo of GAMA",
#                     description="GAMA is a novel Large Large Audio-Language Model that is capable of understanding audio inputs and answer any open-ended question about it." + f"<a href='{paper_link}'>{paper_text}</a> " + f"<a href='{link}'>{text}</a> <br>" +
#                     "GAMA is authored by members of the GAMMA Lab at the University of Maryland, College Park and Adobe, USA. <br>" +
#                     "**GAMA is not an ASR model and has limited ability to recognize the speech content. It primarily focuses on perception and understanding of non-speech sounds.**<br>" +
#                     "Input an audio and ask quesions! Audio will be converted to 16kHz and padded or trim to 10 seconds.")
# demo.launch(debug=True, share=True)
