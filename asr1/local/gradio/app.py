import sys
import io, os, stat
import subprocess
import random
from zipfile import ZipFile
import uuid 
import time
import torch
import torchaudio

import gradio as gr
from scipy.io.wavfile import write
import numpy as np
from pydub import AudioSegment
from resemblyzer import preprocess_wav, VoiceEncoder

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_path",
                    default="tts_models/multilingual/multi-dataset/your_tts",
                    type=str)

parser.add_argument("--download",
                    default="true",
                    choices=["true", "false"],
                    type=str)

args = parser.parse_args()

# args
model_path = args.model_path
download = True if args.download == "true" else False
encoder = VoiceEncoder()

# Initilization for a TTS model
if not download:
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    tts = TTS(model_path=model_path, config_path=config_path, progress_bar=True, gpu=False)
else:
    tts = TTS(model_name=model_path, progress_bar=True, gpu=False)

HF_TOKEN = os.environ.get("HF_TOKEN")

# Use never ffmpeg binary for Ubuntu20 to use denoising for microphone input
print("Export newer ffmpeg binary for denoise filter")
script_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(script_dir, 'ffmpeg.zip')
ZipFile(ffmpeg_path).extractall()
print("Make ffmpeg binary executable")
st = os.stat('ffmpeg')
os.chmod('ffmpeg', st.st_mode | stat.S_IEXEC)

# This is for debugging purposes only
DEVICE_ASSERT_DETECTED=0
DEVICE_ASSERT_PROMPT=None
DEVICE_ASSERT_LANG=None

#supported_languages=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn"]
supported_languages=tts.languages
print("supported_languages", supported_languages)

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def predict(prompt, language, audio_file_pth, mic_file_path, use_mic, voice_cleanup, no_lang_auto_detect, agree,):
    if agree == True:
         
        if language not in supported_languages:
            gr.Warning(f"Language you put {language} in is not in is not in our Supported Languages, please choose from dropdown")
                
            return (
                    None,
                    None,
                    None,
                    None,
                ) 
         
        if use_mic == True:
            if mic_file_path is not None:
               speaker_wav=mic_file_path
            else:
                gr.Warning("Please record your voice with Microphone, or uncheck Use Microphone to use reference audios")
                return (
                    None,
                    None,
                    None,
                    None,
                ) 
                
        else:
            speaker_wav=audio_file_pth

        
        # Filtering for microphone input, as it has BG noise, maybe silence in beginning and end
        # This is fast filtering not perfect

        # Apply all on demand
        lowpassfilter=denoise=trim=loudness=True
        
        if lowpassfilter:
            lowpass_highpass="lowpass=8000,highpass=75," 
        else:
            lowpass_highpass=""

        if trim:
            # better to remove silence in beginning and end for microphone
            trim_silence="areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
        else:
            trim_silence=""
            
        if (voice_cleanup):
            try:
                out_filename = speaker_wav + str(uuid.uuid4()) + ".wav"  #ffmpeg to know output format
    
                #we will use newer ffmpeg as that has after denoise filter
                shell_command = f"./ffmpeg -y -i {speaker_wav} -af {lowpass_highpass}{trim_silence} {out_filename}".split(" ")
    
                command_result = subprocess.run([item for item in shell_command], capture_output=False,text=True, check=True)
                speaker_wav=out_filename
                print("Filtered microphone input")
            except subprocess.CalledProcessError:
                # There was an error - command exited with non-zero code
                print("Error: failed filtering, use original microphone input")
        else:
            speaker_wav=speaker_wav

        if len(prompt)<2:
            gr.Warning("Please give a longer prompt text")
            return (
                    None,
                    None,
                    None,
                    None,
                )
        if len(prompt)>200:
            gr.Warning("Text length limited to 200 characters for this demo, please try shorter text. You can clone this space and edit code for your own usage")
            return (
                    None,
                    None,
                    None,
                    None,
                )  
        global DEVICE_ASSERT_DETECTED
        if DEVICE_ASSERT_DETECTED:
            global DEVICE_ASSERT_PROMPT
            global DEVICE_ASSERT_LANG
            #It will likely never come here as we restart space on first unrecoverable error now
            print(f"Unrecoverable exception caused by language:{DEVICE_ASSERT_LANG} prompt:{DEVICE_ASSERT_PROMPT}")
            
        try:   
            metrics_text=""
            t_latent=time.time()
            
            latent_calculation_time = time.time() - t_latent 
            wav_chunks = []
    
            print("I: Generating new audio...")
            t0 = time.time()
            # TTS
            tts.tts_to_file(prompt, speaker_wav=speaker_wav, language=language, file_path=os.path.join(script_dir, "output.wav"))
            inference_time = time.time() - t0
            print(f"I: Time to generate audio: {round(inference_time*1000)} milliseconds")
            metrics_text+=f"Time to generate audio: {round(inference_time*1000)} milliseconds\n"
            # SECS
            embed1 = encoder.embed_utterance(preprocess_wav(speaker_wav))
            embed2 = encoder.embed_utterance(preprocess_wav(os.path.join(script_dir, "output.wav")))
            cos_sim = cosine_similarity(embed1, embed2)
            print(f"I: SECS: {cos_sim}")
            metrics_text+=f"SECS: {cos_sim}\n"
            #real_time_factor= (time.time() - t0) / out['wav'].shape[-1] * 24000
            #print(f"Real-time factor (RTF): {real_time_factor}")
            #metrics_text+=f"Real-time factor (RTF): {real_time_factor:.2f}\n" 
        except RuntimeError as e :
            if "device-side assert" in str(e):
                # cannot do anything on cuda device side error, need tor estart
                print(f"Exit due to: Unrecoverable exception caused by language:{language} prompt:{prompt}", flush=True)
                gr.Warning("Unhandled Exception encounter, please retry in a minute")
                print("Cuda device-assert Runtime encountered need restart")
                if not DEVICE_ASSERT_DETECTED:
                    DEVICE_ASSERT_DETECTED=1
                    DEVICE_ASSERT_PROMPT=prompt
                    DEVICE_ASSERT_LANG=language

                
            else:
                print("RuntimeError: non device-side assert error:", str(e))
                raise e
        return (
            gr.make_waveform(
                audio=os.path.join(script_dir, "output.wav"),
            ),
            os.path.join(script_dir, "output.wav"),
            metrics_text,
            speaker_wav,
        )
    else:
        gr.Warning("Please accept the Terms & Condition!")
        return (
                None,
                None,
                None,
                None,
            ) 


title = "Coqui🐸"

description = """
<div>
    <a style="display:inline-block" href='https://github.com/coqui-ai/TTS'><img src='https://img.shields.io/github/stars/coqui-ai/TTS?style=social' /></a>
    <a style='display:inline-block' href='https://discord.gg/5eXr5seRrv'><img src='https://discord.com/api/guilds/1037326658807533628/widget.png?style=shield' /></a>
</div>

<br/> This is the same model that powers our creator application <a href="https://coqui.ai">Coqui Studio</a> as well as the <a href="https://docs.coqui.ai">Coqui API</a>. In production we apply modifications to make low-latency streaming possible.
<br/> Leave a star on the Github <a href="https://github.com/coqui-ai/TTS">🐸TTS</a>, where our open-source inference and training code lives.
<br/>
<p>For faster inference without waiting in the queue, you should duplicate this space and upgrade to GPU via the settings.
    <br/>
</p>
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=0d00920c-8cc9-4bf3-90f2-a615797e5f59" />
"""

article = """
<div style='margin:20px auto;'>
<p>By using this demo you agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml</p>
</div>
"""

examples = [
    [
        "Once when I was six years old I saw a magnificent picture",
        "en",
        os.path.join(script_dir, "examples/female.wav"),
        None,
        False,
        False,
        False,
        True,

    ],
]



gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better. Up to 200 text characters.",
            value="Hi there, I'm your new voice clone. Try your best to upload quality audio",
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "en",
            ],
            max_choices=1,
            value="en",
        ),
        gr.Audio(
            label="Reference Audio",
            info="Click on the ✎ button to upload your own target speaker audio",
            type="filepath",
            value=os.path.join(script_dir, "examples/female.wav"),
        ),
        gr.Audio(source="microphone",
                 type="filepath",
                 info="Use your microphone to record audio",
                 label="Use Microphone for Reference"),
        gr.Checkbox(label="Use Microphone",
                    value=False,
                    info="Notice: Microphone input may not work properly under traffic",),
        gr.Checkbox(label="Cleanup Reference Voice",
                    value=False,
                    info="This check can improve output if your microphone or reference voice is noisy",
                    ),
        gr.Checkbox(label="Do not use language auto-detect",
                    value=False,
                    info="Check to disable language auto-detection",),
        gr.Checkbox(
            label="Agree",
            value=False,
            info="I agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml",
        ),

        
    ],
    outputs=[
        gr.Video(label="Waveform Visual"),
        gr.Audio(label="Synthesised Audio",autoplay=True),
        gr.Text(label="Metrics"),
        gr.Audio(label="Reference Audio Used"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
).queue().launch(debug=True, share=True)
