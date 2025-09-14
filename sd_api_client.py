#!/usr/bin/env python3
"""
Stable Diffusion WebUI API Client
Sends requests to the API, processes responses, and saves images locally
"""

import requests
import base64
import json
import os
import time
from datetime import datetime
from PIL import Image, PngImagePlugin
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union
import hashlib

class StableDiffusionAPIClient:
    def __init__(self, base_url: str = "http://localhost:7861", output_dir: str = "api_outputs"):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the Stable Diffusion WebUI API
            output_dir: Directory to save generated images
        """
        self.base_url = base_url.rstrip('/')
        self.output_dir = output_dir
        self.session = requests.Session()
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
    def generate_image(self, payload: Dict) -> Dict:
        """
        Send image generation request to the API
        
        Args:
            payload: Request payload dictionary
            
        Returns:
            API response dictionary
        """
        url = f"{self.base_url}/sdapi/v1/txt2img"
        
        print(f"Sending request to: {url}")
        print(f"Payload preview: {payload.get('prompt', '')[:100]}...")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                headers={
                    'accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                timeout=300  # 5 minute timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            raise
    
    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """
        Decode base64 string to PIL Image
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            PIL Image object
        """
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            return image
            
        except Exception as e:
            print(f"Failed to decode base64 image: {e}")
            raise
    
    def generate_filename(self, payload: Dict, index: int = 0, seed: str = None) -> str:
        """
        Generate a descriptive filename for the saved image
        
        Args:
            payload: Original request payload
            index: Image index in batch
            seed: Seed used for generation
            
        Returns:
            Generated filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a short hash of the prompt for uniqueness
        prompt_hash = hashlib.md5(payload.get('prompt', '').encode()).hexdigest()[:8]
        
        # Extract key parameters
        steps = payload.get('steps', 'unknown')
        cfg_scale = payload.get('cfg_scale', 'unknown')
        sampler = payload.get('sampler_name', 'unknown').replace(' ', '_')
        
        # Build filename
        filename_parts = [
            timestamp,
            f"seed_{seed}" if seed else "seed_auto",
            f"steps_{steps}",
            f"cfg_{cfg_scale}",
            f"sampler_{sampler}",
            f"hash_{prompt_hash}"
        ]
        
        if index > 0:
            filename_parts.append(f"batch_{index:02d}")
        
        filename = "_".join(str(part) for part in filename_parts)
        return f"{filename}.png"
    
    def save_image_with_metadata(self, image: Image.Image, filename: str, 
                                payload: Dict, response_info: str) -> str:
        """
        Save image with embedded metadata
        
        Args:
            image: PIL Image to save
            filename: Filename for the image
            payload: Original request payload
            response_info: Info string from API response
            
        Returns:
            Full path of saved image
        """
        filepath = os.path.join(self.output_dir, "images", filename)
        
        # Add metadata to PNG
        pnginfo = PngImagePlugin.PngInfo()
        
        # Add generation parameters
        pnginfo.add_text("parameters", response_info)
        pnginfo.add_text("prompt", payload.get('prompt', ''))
        pnginfo.add_text("negative_prompt", payload.get('negative_prompt', ''))
        pnginfo.add_text("steps", str(payload.get('steps', '')))
        pnginfo.add_text("cfg_scale", str(payload.get('cfg_scale', '')))
        pnginfo.add_text("sampler_name", payload.get('sampler_name', ''))
        pnginfo.add_text("width", str(payload.get('width', '')))
        pnginfo.add_text("height", str(payload.get('height', '')))
        pnginfo.add_text("script_name", payload.get('script_name', ''))
        pnginfo.add_text("script_args", json.dumps(payload.get('script_args', [])))
        pnginfo.add_text("generation_time", datetime.now().isoformat())
        
        # Save image with metadata
        image.save(filepath, "PNG", pnginfo=pnginfo)
        print(f"Saved image: {filepath}")
        
        return filepath
    
    def save_metadata_json(self, filename: str, payload: Dict, response: Dict) -> str:
        """
        Save detailed metadata as JSON file
        
        Args:
            filename: Base filename (without extension)
            payload: Original request payload
            response: Complete API response
            
        Returns:
            Full path of saved JSON file
        """
        json_filename = filename.replace('.png', '.json')
        json_filepath = os.path.join(self.output_dir, "metadata", json_filename)
        
        metadata = {
            "generation_time": datetime.now().isoformat(),
            "request_payload": payload,
            "api_response": {
                "info": response.get('info', ''),
                "parameters": response.get('parameters', {}),
                "image_count": len(response.get('images', []))
            }
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved metadata: {json_filepath}")
        return json_filepath
    
    def process_response(self, response: Dict, payload: Dict) -> List[str]:
        """
        Process API response and save all images
        
        Args:
            response: API response dictionary
            payload: Original request payload
            
        Returns:
            List of saved image file paths
        """
        images = response.get('images', [])
        if not images:
            print("No images in response")
            return []
        
        print(f"Processing {len(images)} images from response")
        
        saved_files = []
        response_info = response.get('info', '')
        
        # Extract seed from response info if available
        seed = "unknown"
        if response_info:
            try:
                # Try to extract seed from info string
                import re
                seed_match = re.search(r'Seed: (\d+)', response_info)
                if seed_match:
                    seed = seed_match.group(1)
            except:
                pass
        
        for i, img_b64 in enumerate(images):
            try:
                # Decode image
                image = self.decode_base64_image(img_b64)
                
                # Generate filename
                filename = self.generate_filename(payload, i, seed)
                
                # Save image with metadata
                filepath = self.save_image_with_metadata(image, filename, payload, response_info)
                saved_files.append(filepath)
                
                # Save JSON metadata for first image
                if i == 0:
                    self.save_metadata_json(filename, payload, response)
                
            except Exception as e:
                print(f"Failed to process image {i}: {e}")
                continue
        
        return saved_files
    
    def generate_and_save(self, payload: Dict) -> List[str]:
        """
        Complete workflow: generate image and save results
        
        Args:
            payload: Request payload dictionary
            
        Returns:
            List of saved image file paths
        """
        print("=" * 60)
        print("STABLE DIFFUSION API CLIENT")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Send API request
            response = self.generate_image(payload)
            
            # Process and save images
            saved_files = self.process_response(response, payload)
            
            end_time = time.time()
            print(f"\n✅ Generation completed in {end_time - start_time:.2f} seconds")
            print(f"📁 Saved {len(saved_files)} images to: {self.output_dir}")
            
            return saved_files
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return []

def create_wedding_dress_payload(clothing_tags: str = "formal outfit"):
    """Create the outfit generation payload with specified clothing tags"""
    return {
        "prompt": f"ultra-realistic photo of a young man, wearing hoodie, blue jeans, denim jacket and slippers , walking in a green park during daytime, natural sunlight, cinematic photography, depth of field, highly detailed, 8k resolution",
        "negative_prompt": "blurry, low resolution, distorted face, extra limbs, deformed hands, cartoon, anime, painting, unrealistic proportions, bad anatomy, watermark, text, noise, overexposed, underexposed",
        "styles": [],
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "sampler_name": "DPM++ 2M",
        "scheduler": "Automatic",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 30,
        "cfg_scale": 8.0,
        "width": 768,
        "height": 1024,
        "restore_faces": False,
        "tiling": False,
        "do_not_save_samples": False,
        "do_not_save_grid": False,
        "eta": 0,
        "denoising_strength": 0,
        "s_min_uncond": 0,
        "s_churn": 0,
        "s_tmax": 0,
        "s_tmin": 0,
        "s_noise": 0,
        "override_settings": {},
        "override_settings_restore_afterwards": True,
        "refiner_checkpoint": "",
        "refiner_switch_at": 0,
        "disable_extra_networks": False,
        "firstpass_image": "",
        "comments": {},
        "enable_hr": False,
        "firstphase_width": 0,
        "firstphase_height": 0,
        "hr_scale": 2,
        "hr_upscaler": "",
        "hr_second_pass_steps": 0,
        "hr_resize_x": 0,
        "hr_resize_y": 0,
        "hr_checkpoint_name": "",
        "hr_sampler_name": "",
        "hr_scheduler": "",
        "hr_prompt": "",
        "hr_negative_prompt": "",
        "force_task_id": "",
        "sampler_index": "Euler",
        "script_name": "accelerate with qualcomm ai runtime",
        "script_args": ["Stable-Diffusion-1.5", False, ""],
        "send_images": True,
        "save_images": True,
        "alwayson_scripts": {},
        "infotext": ""
    }

def main():
    """Main function to run the client"""
    # Configuration
    API_URL = "http://localhost:7861"
    OUTPUT_DIR = "generated_images"
    
    # Initialize client
    client = StableDiffusionAPIClient(base_url=API_URL, output_dir=OUTPUT_DIR)
    
    # Create payload
    payload = create_wedding_dress_payload()
    
    # Generate and save images
    saved_files = client.generate_and_save(payload)
    
    if saved_files:
        print("\n🎉 SUCCESS!")
        print("Generated images saved to:")
        for filepath in saved_files:
            print(f"  • {filepath}")
    else:
        print("\n❌ No images were generated or saved")

if __name__ == "__main__":
    main()
