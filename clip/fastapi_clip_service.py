# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from typing import Dict, List, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import os
from dataclasses import dataclass
from enum import Enum

# Import your existing CLIP ONNX module
from clip_onnx_qnn import initialize_clip_onnx, ClipAppONNX

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define clothing categories and tags
CATEGORIES = ["top", "bottom", "dress", "outerwear", "footwear", "accessory"]

SUBCATEGORIES = {
    "top": ["t-shirt", "shirt", "blouse", "sweater", "hoodie", "tank top", "polo"],
    "bottom": ["jeans", "trousers", "shorts", "skirt", "leggings"],
    "dress": ["dress", "jumpsuit"],
    "outerwear": ["jacket", "coat", "blazer", "cardigan"],
    "footwear": ["sneakers", "boots", "heels", "sandals", "loafers"],
    "accessory": ["hat", "bag", "belt", "scarf"],
}

TAG_VALUES = {
    "color": ["black", "white", "gray", "beige", "brown", "red", "blue", 
              "green", "yellow", "pink", "purple", "orange"],
    "style": ["casual", "formal", "semi-formal", "sporty", "party", "cozy"],
    "season": ["summer", "winter", "all-season"],
    "pattern": ["solid", "striped", "checked", "floral", "graphic", "denim", "leather"],
}

# Configuration
@dataclass
class Config:
    onnx_model_path: str = "models/clip_model.onnx"
    image_encoder_path: Optional[str] = None
    text_encoder_path: Optional[str] = None
    model_name: str = "openai/clip-vit-base-patch32"
    use_cpu_fallback: bool = True
    max_workers: int = 4
    similarity_threshold: float = 20.0  # Minimum similarity score to consider
    top_k_tags: int = 3  # Number of top tags to return per category
    allowed_directories: List[str] = None  # Directories allowed for image path access
    
    def __post_init__(self):
        # Set default allowed directories
        if self.allowed_directories is None:
            self.allowed_directories = [
                "data",
                "sample_images", 
                "my_wardrobe",
                "uploads",
                "images"
            ]

config = Config()

# Initialize FastAPI app
app = FastAPI(
    title="CLIP Clothing Tagger API",
    description="API for tagging clothing items using CLIP model on Qualcomm NPU",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=config.max_workers)

# Global model instance
clip_model: Optional[ClipAppONNX] = None


class ClothingTagger:
    """Main class for clothing tagging using CLIP"""
    
    def __init__(self, clip_app: ClipAppONNX):
        self.clip_app = clip_app
        self.all_prompts = self._generate_prompts()
        
    def _generate_prompts(self) -> Dict[str, List[str]]:
        """Generate all text prompts for classification"""
        prompts = {}
        
        # Main category prompts - more specific descriptions
        prompts["main_category"] = [
            "a photo of a top clothing item like shirt, t-shirt, blouse, sweater, or hoodie",
            "a photo of a bottom clothing item like pants, jeans, trousers, shorts, or skirt", 
            "a photo of a dress or jumpsuit",
            "a photo of outerwear like jacket, coat, blazer, or cardigan",
            "a photo of footwear like shoes, sneakers, boots, heels, or sandals",
            "a photo of an accessory like hat, bag, belt, or scarf"
        ]
        
        # Subcategory prompts for each main category
        for category, items in SUBCATEGORIES.items():
            prompts[f"subcategory_{category}"] = [f"a photo of a {item}" for item in items]
        
        # Color prompts - more specific
        prompts["color"] = [f"a photo of {color} colored clothing" for color in TAG_VALUES["color"]]
        
        # Style prompts
        prompts["style"] = [f"a photo of {style} style clothing" for style in TAG_VALUES["style"]]
        
        # Season prompts
        prompts["season"] = [f"a photo of {season} clothing" for season in TAG_VALUES["season"]]
        
        # Pattern prompts
        prompts["pattern"] = [f"a photo of clothing with {pattern} pattern" for pattern in TAG_VALUES["pattern"]]
        
        return prompts
    
    def _process_similarities(self, similarities: np.ndarray, prompts: List[str], 
                            threshold: float = 20.0) -> Dict[str, Any]:
        """Process similarity scores and return top match only"""
        # Get the best match
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        
        if best_score >= threshold:
            # Extract the actual tag from the prompt
            tag = prompts[best_idx].replace("a photo of a ", "").replace("a photo of ", "")
            tag = tag.replace(" colored clothing", "").replace(" style clothing", "")
            tag = tag.replace(" clothing", "").replace("clothing with ", "").replace(" pattern", "")
            return {
                "tag": tag,
                "confidence": round(best_score, 2)
            }
        
        return None
    
    def tag_image(self, image: Image.Image) -> Dict[str, Any]:
        """Tag a single image with clothing attributes"""
        tags = {}
        
        # Step 1: Determine main category
        main_category_prompts = self.all_prompts["main_category"]
        similarities = self.clip_app.predict_similarity([image], main_category_prompts)
        similarities = similarities.flatten()
        
        # Find the best matching main category
        best_category_idx = np.argmax(similarities)
        best_category_score = float(similarities[best_category_idx])
        
        if best_category_score >= config.similarity_threshold:
            main_category = CATEGORIES[best_category_idx]
            tags["category"] = main_category
            
            # Step 2: Determine subcategory within the identified main category
            subcategory_key = f"subcategory_{main_category}"
            if subcategory_key in self.all_prompts:
                subcategory_prompts = self.all_prompts[subcategory_key]
                sub_similarities = self.clip_app.predict_similarity([image], subcategory_prompts)
                sub_similarities = sub_similarities.flatten()
                
                # Get best subcategory match
                best_sub_idx = np.argmax(sub_similarities)
                best_sub_score = float(sub_similarities[best_sub_idx])
                
                if best_sub_score >= config.similarity_threshold:
                    subcategory = SUBCATEGORIES[main_category][best_sub_idx]
                    tags["subcategory"] = subcategory
        
        # Step 3: Process other attributes (color, style, season, pattern)
        for prompt_category in ["color", "style", "season", "pattern"]:
            if prompt_category in self.all_prompts:
                prompts = self.all_prompts[prompt_category]
                similarities = self.clip_app.predict_similarity([image], prompts)
                similarities = similarities.flatten()
                
                top_match = self._process_similarities(
                    similarities, prompts, 
                    threshold=config.similarity_threshold
                )
                if top_match:
                    tags[prompt_category] = top_match["tag"]
        
        return tags
    
    def batch_tag_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Tag multiple images efficiently"""
        results = []
        for image in images:
            tags = self.tag_image(image)
            results.append(tags)
        return results


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global clip_model
    
    try:
        logger.info("Initializing CLIP model...")
        
        # Check if model exists
        if not os.path.exists(config.onnx_model_path):
            logger.error(f"Model not found at {config.onnx_model_path}")
            raise FileNotFoundError(f"Model not found at {config.onnx_model_path}")
        
        # Initialize model
        if config.image_encoder_path and config.text_encoder_path:
            clip_model = initialize_clip_onnx(
                image_encoder_path=config.image_encoder_path,
                text_encoder_path=config.text_encoder_path,
                model_name=config.model_name,
                use_cpu_fallback=config.use_cpu_fallback
            )
        else:
            clip_model = initialize_clip_onnx(
                model_path=config.onnx_model_path,
                model_name=config.model_name,
                use_cpu_fallback=config.use_cpu_fallback
            )
        
        logger.info("CLIP model initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CLIP Clothing Tagger API",
        "status": "running",
        "model_loaded": clip_model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": clip_model is not None
    }


@app.post("/tag")
async def tag_clothing(
    file: UploadFile = File(None),
    image_path: str = None
):
    """
    Tag a single clothing image
    
    Args:
        file: Image file upload (JPEG, PNG, etc.)
        image_path: Path to image on server filesystem
    
    Returns:
        JSON with clothing tags and confidence scores
    """
    if clip_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Validate input - need either file or path
    if not file and not image_path:
        raise HTTPException(
            status_code=400, 
            detail="Either 'file' upload or 'image_path' must be provided"
        )
    
    if file and image_path:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'file' upload or 'image_path', not both"
        )
    
    try:
        # Load image based on input type
        if file:
            # Handle file upload
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            filename = file.filename
        else:
            # Handle file path
            if not os.path.exists(image_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Image not found at path: {image_path}"
                )
            
            # Security check - prevent directory traversal
            abs_path = os.path.abspath(image_path)
            allowed_dirs = [os.path.abspath(d) for d in config.allowed_directories]
            
            if not any(abs_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                raise HTTPException(
                    status_code=403,
                    detail="Access to this path is not allowed"
                )
            
            image = Image.open(image_path).convert('RGB')
            filename = os.path.basename(image_path)
        
        # Create tagger instance
        tagger = ClothingTagger(clip_model)
        
        # Process image in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        tags = await loop.run_in_executor(executor, tagger.tag_image, image)
        
        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "source": "upload" if file else "path",
            "tags": tags
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/tag-batch")
async def tag_clothing_batch(
    files: List[UploadFile] = File(None),
    image_paths: str = None
):
    """
    Tag multiple clothing images in batch
    
    Args:
        files: List of image file uploads
        image_paths: Comma-separated list of image paths on server
    
    Returns:
        JSON with tags for each image
    """
    if clip_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Validate input
    if not files and not image_paths:
        raise HTTPException(
            status_code=400,
            detail="Either 'files' upload or 'image_paths' must be provided"
        )
    
    if files and image_paths:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'files' upload or 'image_paths', not both"
        )
    
    try:
        images = []
        filenames = []
        
        if files:
            # Process uploaded files
            if len(files) > 10:
                raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
            
            for file in files:
                if file.filename:  # Check if file was actually uploaded
                    contents = await file.read()
                    image = Image.open(io.BytesIO(contents)).convert('RGB')
                    images.append(image)
                    filenames.append(file.filename)
        else:
            # Process paths
            paths = [p.strip() for p in image_paths.split(",")]
            
            if len(paths) > 10:
                raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
            
            # Security check for allowed directories
            allowed_dirs = [os.path.abspath(d) for d in config.allowed_directories]
            
            for path in paths:
                if not os.path.exists(path):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Image not found: {path}"
                    )
                
                abs_path = os.path.abspath(path)
                if not any(abs_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Access denied for path: {path}"
                    )
                
                image = Image.open(path).convert('RGB')
                images.append(image)
                filenames.append(os.path.basename(path))
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Create tagger and process
        tagger = ClothingTagger(clip_model)
        
        # Process in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, tagger.batch_tag_images, images)
        
        # Format response
        response = {
            "status": "success",
            "source": "upload" if files else "paths",
            "results": [
                {"filename": filename, "tags": tags}
                for filename, tags in zip(filenames, results)
            ]
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing batch: {str(e)}")


@app.get("/categories")
async def get_categories():
    """Get all available clothing categories and tags"""
    return {
        "categories": CATEGORIES,
        "subcategories": SUBCATEGORIES,
        "tag_values": TAG_VALUES
    }


@app.post("/tag-with-custom-labels")
async def tag_with_custom_labels(
    file: UploadFile = File(None),
    image_path: str = None,
    custom_labels: str = None
):
    """
    Tag clothing with custom labels
    
    Args:
        file: Image file upload
        image_path: Path to image on server
        custom_labels: Comma-separated custom labels to check
    
    Returns:
        Similarity scores for custom labels
    """
    if clip_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if not custom_labels:
        raise HTTPException(status_code=400, detail="custom_labels parameter required")
    
    # Validate input - need either file or path
    if not file and not image_path:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' upload or 'image_path' must be provided"
        )
    
    if file and image_path:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'file' upload or 'image_path', not both"
        )
    
    try:
        # Parse custom labels
        labels = [label.strip() for label in custom_labels.split(",")]
        prompts = [f"a photo of a {label}" for label in labels]
        
        # Load image
        if file:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            filename = file.filename
        else:
            # Validate path
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
            
            # Security check
            abs_path = os.path.abspath(image_path)
            allowed_dirs = [os.path.abspath(d) for d in config.allowed_directories]
            
            if not any(abs_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                raise HTTPException(status_code=403, detail="Access to this path is not allowed")
            
            image = Image.open(image_path).convert('RGB')
            filename = os.path.basename(image_path)
        
        # Get similarities
        loop = asyncio.get_event_loop()
        similarities = await loop.run_in_executor(
            executor, 
            clip_model.predict_similarity, 
            [image], 
            prompts
        )
        similarities = similarities.flatten()
        
        # Format results
        results = [
            {"label": label, "confidence": round(float(score), 2)}
            for label, score in zip(labels, similarities)
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "source": "upload" if file else "path",
            "results": results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error with custom labels: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing: {str(e)}")


@app.post("/tag-folder")
async def tag_folder(folder_path: str):
    """
    Tag all images in a folder
    
    Args:
        folder_path: Path to folder containing images
    
    Returns:
        JSON with tags for all images in the folder
    """
    if clip_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Validate folder path
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {folder_path}")
        
        # Security check
        abs_folder = os.path.abspath(folder_path)
        allowed_dirs = [os.path.abspath(d) for d in config.allowed_directories]
        
        if not any(abs_folder.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Allowed directories: {config.allowed_directories}"
            )
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_files = []
        
        for file in Path(folder_path).iterdir():
            if file.suffix.lower() in image_extensions:
                image_files.append(str(file))
        
        if not image_files:
            raise HTTPException(status_code=404, detail="No images found in folder")
        
        if len(image_files) > 50:
            raise HTTPException(
                status_code=400,
                detail=f"Too many images ({len(image_files)}). Maximum 50 images allowed."
            )
        
        # Process images
        tagger = ClothingTagger(clip_model)
        results = []
        
        for img_path in image_files:
            try:
                image = Image.open(img_path).convert('RGB')
                loop = asyncio.get_event_loop()
                tags = await loop.run_in_executor(executor, tagger.tag_image, image)
                
                results.append({
                    "filename": os.path.basename(img_path),
                    "path": img_path,
                    "tags": tags
                })
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append({
                    "filename": os.path.basename(img_path),
                    "path": img_path,
                    "error": str(e)
                })
        
        return JSONResponse(content={
            "status": "success",
            "folder": folder_path,
            "total_images": len(image_files),
            "processed": len([r for r in results if "tags" in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing folder: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing folder: {str(e)}")


# Main execution block - FIXED INDENTATION
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CLIP Clothing Tagger API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", default="models/clip_model.onnx", help="Path to ONNX model")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Update config
    config.onnx_model_path = args.model_path
    
    # Run server
    uvicorn.run(
        "fastapi_clip_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )