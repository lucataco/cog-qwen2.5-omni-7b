# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, BaseModel
import os
import time
import torch
import subprocess
import soundfile as sf
from typing import Optional
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/qwen/Qwen2.5-Omni-7B/model.tar"

def download_weights(url, dest):
    start = time.time()
    print(f"downloading url: {url}")
    print(f"downloading to: {dest}")
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print(f"downloading took: {time.time() - start}")

class Output(BaseModel):
    text: str
    voice: Optional[Path]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.model = Qwen2_5OmniModel.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            enable_audio_output=True,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_CACHE)

    def predict(
        self,
        prompt: str = Input(description="Text prompt for the model", default=None),
        image: Optional[Path] = Input(description="Optional image input", default=None),
        audio: Optional[Path] = Input(description="Optional audio input", default=None),
        video: Optional[Path] = Input(description="Optional video input", default=None),
        system_prompt: str = Input(
            description="System prompt for the model",
            default="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        ),
        use_audio_in_video: bool = Input(
            description="Whether to use audio in video",
            default=True
        ),
        voice_type: str = Input(
            description="Voice type for audio output",
            default="Chelsie",
            choices=["Chelsie", "Ethan"]
        ),
        generate_audio: bool = Input(
            description="Whether to generate audio output",
            default=True
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        # Prepare the conversation
        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        
        user_content = []
        
        # Add media content if provided
        if image is not None:
            user_content.append({"type": "image", "image": str(image)})
        
        if audio is not None:
            user_content.append({"type": "audio", "audio": str(audio)})
            
        if video is not None:
            user_content.append({"type": "video", "video": str(video)})
            
        # Add text prompt if it's not empty
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        
        # If user_content is empty, just use the prompt as a string
        if not user_content:
            conversation.append({"role": "user", "content": prompt})
        else:
            conversation.append({"role": "user", "content": user_content})
        
        # Prepare inputs
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        
        inputs = self.processor(
            text=text, 
            audios=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=use_audio_in_video
        )
        
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        # Generate output
        if generate_audio:
            text_ids, audio_output = self.model.generate(
                **inputs, 
                use_audio_in_video=use_audio_in_video,
                spk=voice_type,
                return_audio=True
            )
            
            # Save audio to a temporary file
            audio_path = "output.wav"
            sf.write(
                audio_path,
                audio_output.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )
            
            # Decode text and remove system prompt from output
            text_output = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # Clean the text output by removing system prompt and user prompt markers
            text_output = self.clean_output_text(text_output)
            
            return Output(text=text_output, voice=Path(audio_path))
        else:
            text_ids = self.model.generate(
                **inputs, 
                use_audio_in_video=use_audio_in_video,
                return_audio=False
            )
            
            # Decode text and remove system prompt from output
            text_output = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # Clean the text output by removing system prompt and user prompt markers
            text_output = self.clean_output_text(text_output)
            
            return Output(text=text_output, voice=None)

    def clean_output_text(self, text):
        """Remove system prompt and user markers from the output text"""
        # Remove everything before "assistant" if it exists
        if "assistant" in text.lower():
            text = text[text.lower().find("assistant") + len("assistant"):].strip()
        
        # Remove "system" and the following system prompt if present
        if "system" in text.lower():
            system_start = text.lower().find("system")
            user_start = text.lower().find("user", system_start)
            if user_start != -1:
                # Remove everything between "system" and "user"
                text = text[:system_start] + text[user_start:]
            
        # Remove "user" markers if they exist
        if "user" in text.lower():
            user_start = text.lower().find("user")
            assistant_start = text.lower().find("assistant", user_start)
            if assistant_start != -1:
                # Remove everything between "user" and "assistant"
                text = text[:user_start] + text[assistant_start + len("assistant"):]
            else:
                # Remove "user" and everything after it if no "assistant" is found
                text = text[:user_start].strip()
                
        return text.strip()
