#!/usr/bin/env python3
"""
Simple Stable Diffusion API Client
Quick and easy image generation and saving
"""

import requests
import base64
import json
import os
from PIL import Image
from io import BytesIO
from datetime import datetime

def send_request_and_save(api_url="http://localhost:7861", output_dir="outputs"):
    """Send the exact curl request and save the result"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Your exact payload
    payload = {
        "prompt": "elegant white wedding dress, flowing silk fabric, intricate lace details, cathedral train, off-shoulder design, pearl embellishments, soft romantic lighting, professional fashion photography, high fashion, detailed fabric texture, pristine white, bridal portrait, studio lighting, 8k resolution, hyperrealistic",
        "negative_prompt": "low quality, blurry, distorted, amateur photography, poor lighting, wrinkled fabric, stained, torn, inappropriate, nsfw, extra limbs, deformed hands",
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
    
    print("🚀 Sending request to Stable Diffusion API...")
    print(f"📡 URL: {api_url}/sdapi/v1/txt2img")
    
    try:
        # Send POST request
        response = requests.post(
            f"{api_url}/sdapi/v1/txt2img",
            json=payload,
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            timeout=300
        )
        
        response.raise_for_status()
        result = response.json()
        
        print("✅ Request successful!")
        print(f"📊 Response info: {result.get('info', 'No info available')[:100]}...")
        
        # Process images
        images = result.get('images', [])
        if not images:
            print("❌ No images in response")
            return
        
        print(f"🖼️  Found {len(images)} images in response")
        
        # Save each image
        for i, img_b64 in enumerate(images):
            # Decode base64
            img_data = base64.b64decode(img_b64)
            img = Image.open(BytesIO(img_data))
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wedding_dress_{timestamp}_{i:02d}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            img.save(filepath)
            print(f"💾 Saved: {filepath}")
            print(f"   Size: {img.size[0]}x{img.size[1]} pixels")
        
        # Save response metadata
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"📄 Saved metadata: {metadata_file}")
        
        print("\n🎉 All done! Check your output directory for the images.")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error processing response: {e}")

if __name__ == "__main__":
    # Run the client
    send_request_and_save()
