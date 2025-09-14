# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from pathlib import Path
from typing import Union, List, Sequence, Callable, Optional, Dict, Any
import warnings
import time


def get_onnxruntime_session_cpu(model_path: str):
    """Create ONNX Runtime session with CPU provider for testing"""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )
    return session


def get_onnxruntime_session_with_qnn_ep(model_path: str, use_cpu_fallback: bool = True):
    """
    Create ONNX Runtime session with QNN Execution Provider for Qualcomm NPU.
    Falls back to CPU if QNN is not available or fails.
    """
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    
    try:
        # Check if QNN provider is available
        available_providers = ort.get_available_providers()
        
        if "QNNExecutionProvider" not in available_providers:
            warnings.warn("QNNExecutionProvider not available. Using CPU fallback.")
            if use_cpu_fallback:
                return get_onnxruntime_session_cpu(model_path)
            raise RuntimeError("QNNExecutionProvider not available")
        
        # Try to create session with QNN EP
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=[
                ("QNNExecutionProvider", {
                    "backend_path": "QnnHtp.dll",
                    "htp_performance_mode": "high_performance",
                    "enable_htp_fp16_precision": "1", 
                }),
                "CPUExecutionProvider"  # Fallback
            ]
        )
        
        # Check which provider is actually being used
        actual_provider = session.get_providers()[0]
        print(f"Using provider: {actual_provider}")
        
        return session
        
    except Exception as e:
        warnings.warn(f"Failed to create QNN session: {e}")
        if use_cpu_fallback:
            print("Falling back to CPU provider...")
            return get_onnxruntime_session_cpu(model_path)
        raise


class ONNXClipWrapper:
    """
    Wrapper for CLIP ONNX model to run on Qualcomm NPU using QNN EP.
    Handles both combined and separate encoder models.
    """
    
    def __init__(
        self, 
        model_path: str = None,
        image_encoder_path: str = None,
        text_encoder_path: str = None,
        use_cpu_fallback: bool = True
    ):
        """
        Initialize the CLIP model wrapper.
        
        Args:
            model_path: Path to combined CLIP ONNX model
            image_encoder_path: Path to image encoder ONNX model
            text_encoder_path: Path to text encoder ONNX model
            use_cpu_fallback: Whether to fall back to CPU if QNN fails
        """
        self.use_separate_encoders = False
        
        if model_path:
            # Single combined model
            self.session = get_onnxruntime_session_with_qnn_ep(model_path, use_cpu_fallback)
            self.image_session = None
            self.text_session = None
        elif image_encoder_path and text_encoder_path:
            # Separate encoder models
            self.use_separate_encoders = True
            self.image_session = get_onnxruntime_session_with_qnn_ep(image_encoder_path, use_cpu_fallback)
            self.text_session = get_onnxruntime_session_with_qnn_ep(text_encoder_path, use_cpu_fallback)
            self.session = None
        else:
            raise ValueError("Must provide either model_path or both image_encoder_path and text_encoder_path")
        
        # Get input/output metadata
        if self.session:
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            print(f"Combined model inputs: {self.input_names}")
            print(f"Combined model outputs: {self.output_names}")
        else:
            self.image_input_names = [inp.name for inp in self.image_session.get_inputs()]
            self.image_output_names = [out.name for out in self.image_session.get_outputs()]
            self.text_input_names = [inp.name for inp in self.text_session.get_inputs()]
            self.text_output_names = [out.name for out in self.text_session.get_outputs()]
            print(f"Image encoder inputs: {self.image_input_names}")
            print(f"Text encoder inputs: {self.text_input_names}")
    
    def encode_image(self, image_tensor: np.ndarray) -> np.ndarray:
        """Encode images to feature vectors"""
        start_time = time.time()
        
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = image_tensor.numpy()
        
        if self.use_separate_encoders:
            # Use separate image encoder
            inputs = {self.image_input_names[0]: image_tensor.astype(np.float32)}
            outputs = self.image_session.run(None, inputs)
            result = outputs[0]
        else:
            # Extract from combined model - need text dummy input
            dummy_text = np.zeros((image_tensor.shape[0], 77), dtype=np.int32)
            result = self._run_combined(image_tensor, dummy_text, return_image_features=True)
        
        end_time = time.time()
        print(f"Image encoding time: {(end_time - start_time)*1000:.2f} ms")
        return result
    
    def encode_text(self, text_tokens: np.ndarray) -> np.ndarray:
        """Encode text to feature vectors"""
        start_time = time.time()
        
        if isinstance(text_tokens, torch.Tensor):
            text_tokens = text_tokens.numpy()
        
        if self.use_separate_encoders:
            # Use separate text encoder
            inputs = {self.text_input_names[0]: text_tokens.astype(np.int64)}
            outputs = self.text_session.run(None, inputs)
            result = outputs[0]
        else:
            # Extract from combined model - need image dummy input
            dummy_image = np.zeros((text_tokens.shape[0], 3, 224, 224), dtype=np.float32)
            result = self._run_combined(dummy_image, text_tokens, return_text_features=True)
        
        end_time = time.time()
        print(f"Text encoding time: {(end_time - start_time)*1000:.2f} ms")
        return result
    
    def _run_combined(
        self, 
        image_tensor: np.ndarray, 
        text_tokens: np.ndarray,
        return_image_features: bool = False,
        return_text_features: bool = False
    ):
        """Run combined model inference"""
        start_time = time.time()
        
        # Convert tensors to numpy if needed
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = image_tensor.numpy()
        if isinstance(text_tokens, torch.Tensor):
            text_tokens = text_tokens.numpy()
        
        inputs = {}
        
        # Map inputs based on name patterns
        for input_name in self.input_names:
            input_lower = input_name.lower()
            if any(x in input_lower for x in ['image', 'pixel', 'visual']):
                inputs[input_name] = image_tensor.astype(np.float32)
            elif any(x in input_lower for x in ['text', 'input_ids', 'token']):
                inputs[input_name] = text_tokens.astype(np.int32)
        
        # Fallback to positional if no pattern match
        if not inputs and len(self.input_names) >= 2:
            inputs[self.input_names[0]] = image_tensor.astype(np.float32)
            inputs[self.input_names[1]] = text_tokens.astype(np.int32)
        
        outputs = self.session.run(None, inputs)
        
        end_time = time.time()
        print(f"Combined model inference time: {(end_time - start_time)*1000:.2f} ms")
        
        # Return appropriate features
        if return_image_features:
            return outputs[0] if len(outputs) > 0 else None
        elif return_text_features:
            return outputs[1] if len(outputs) > 1 else outputs[0]
        else:
            # Return similarity scores or all outputs
            return outputs[0] if outputs else None
    
    def compute_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between image and text features"""
        start_time = time.time()
        
        # Normalize features
        image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
        
        # Compute cosine similarity
        similarity = np.matmul(image_features, text_features.T)
        result = similarity * 100.0  # Scale to percentage
        
        end_time = time.time()
        print(f"Similarity computation time: {(end_time - start_time)*1000:.2f} ms")
        return result
    
    def __call__(self, image_tensor: np.ndarray, text_tokens: np.ndarray):
        """
        Run inference on the CLIP model.
        """
        if self.use_separate_encoders:
            image_features = self.encode_image(image_tensor)
            text_features = self.encode_text(text_tokens)
            return self.compute_similarity(image_features, text_features)
        else:
            return self._run_combined(image_tensor, text_tokens)


class ClipAppONNX:
    """
    Application class for end-to-end CLIP inference using ONNX Runtime with QNN EP.
    """
    
    def __init__(
        self,
        model_path: str = None,
        image_encoder_path: str = None,
        text_encoder_path: str = None,
        text_tokenizer: Callable[[str], torch.Tensor] = None,
        image_preprocessor: Callable[[Image.Image], torch.Tensor] = None,
        use_cpu_fallback: bool = True
    ):
        """
        Initialize the CLIP application.
        
        Args:
            model_path: Path to combined ONNX model
            image_encoder_path: Path to image encoder ONNX
            text_encoder_path: Path to text encoder ONNX
            text_tokenizer: Function to tokenize text
            image_preprocessor: Function to preprocess images
            use_cpu_fallback: Whether to fall back to CPU
        """
        self.model = ONNXClipWrapper(
            model_path=model_path,
            image_encoder_path=image_encoder_path,
            text_encoder_path=text_encoder_path,
            use_cpu_fallback=use_cpu_fallback
        )
        self.text_tokenizer = text_tokenizer
        self.image_preprocessor = image_preprocessor
    
    def predict(self, *args, **kwargs):
        """Alias for predict_similarity"""
        return self.predict_similarity(*args, **kwargs)
    
    def predict_similarity(
        self, 
        images_or_image_paths: Sequence[Union[Image.Image, str, Path]], 
        texts: Sequence[str]
    ) -> np.ndarray:
        """
        Compute similarity between images and texts.
        
        Args:
            images_or_image_paths: List of PIL Images or paths to image files
            texts: List of text strings to compare
            
        Returns:
            Cosine similarities matrix (num_images x num_texts)
        """
        total_start_time = time.time()
        print(f"\nStarting inference for {len(images_or_image_paths)} images and {len(texts)} texts...")
        
        # This model computes similarity directly for each image-text pair
        similarities = []
        
        for img_idx, image_or_path in enumerate(images_or_image_paths):
            image_start_time = time.time()
            print(f"\nProcessing image {img_idx + 1}/{len(images_or_image_paths)}: {image_or_path}")
            
            if isinstance(image_or_path, (str, Path)):
                image = Image.open(image_or_path).convert('RGB')
            else:
                image = image_or_path
            
            # Process single image
            image_tensor = self.image_preprocessor(image)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_sims = []
            for text_idx, text in enumerate(texts):
                text_start_time = time.time()
                print(f"  Processing text {text_idx + 1}/{len(texts)}: {text[:50]}...")
                
                tokens = self.text_tokenizer(text)
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)
                
                # Run model for this image-text pair
                similarity = self.model(image_tensor, tokens)
                if isinstance(similarity, np.ndarray):
                    image_sims.append(similarity.item())
                else:
                    image_sims.append(similarity)
                
                text_end_time = time.time()
                print(f"    Text processing time: {(text_end_time - text_start_time)*1000:.2f} ms")
            
            image_end_time = time.time()
            print(f"  Total image processing time: {(image_end_time - image_start_time)*1000:.2f} ms")
            similarities.append(image_sims)
        
        total_end_time = time.time()
        print(f"\nTotal inference time: {(total_end_time - total_start_time)*1000:.2f} ms")
        print(f"Average time per image-text pair: {(total_end_time - total_start_time)*1000/(len(images_or_image_paths)*len(texts)):.2f} ms")
        
        return np.array(similarities)


def initialize_clip_onnx(
    model_path: str = None,
    image_encoder_path: str = None,
    text_encoder_path: str = None,
    model_name: str = "openai/clip-vit-base-patch32",
    use_cpu_fallback: bool = True
):
    """
    Initialize CLIP model with ONNX Runtime and QNN EP.
    
    Args:
        model_path: Path to combined ONNX model
        image_encoder_path: Path to image encoder ONNX
        text_encoder_path: Path to text encoder ONNX
        model_name: HuggingFace model name for tokenizer/preprocessor
        use_cpu_fallback: Whether to fall back to CPU
        
    Returns:
        ClipAppONNX instance ready for inference
    """
    
    try:
        from transformers import CLIPTokenizer, CLIPImageProcessor
        
        # Initialize tokenizer and preprocessor
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        image_processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Create tokenizer function
        def text_tokenizer_fn(text: str) -> torch.Tensor:
            tokens = tokenizer(
                text, 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="pt"
            )
            return tokens["input_ids"]
        
        # Create image preprocessor function
        def image_preprocessor_fn(image: Image.Image) -> torch.Tensor:
            processed = image_processor(images=image, return_tensors="pt")
            return processed["pixel_values"][0]
        
    except ImportError:
        print("Warning: transformers not installed. Using basic preprocessing.")
        print("Install with: pip install transformers")
        
        # Fallback preprocessing
        from torchvision import transforms
        
        def text_tokenizer_fn(text: str) -> torch.Tensor:
            # Basic placeholder - won't work properly
            tokens = torch.zeros(1, 77, dtype=torch.long)
            warnings.warn("Using dummy tokenizer - install transformers for proper tokenization")
            return tokens
        
        def image_preprocessor_fn(image: Image.Image) -> torch.Tensor:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            return transform(image)
    
    # Create and return the app
    app = ClipAppONNX(
        model_path=model_path,
        image_encoder_path=image_encoder_path,
        text_encoder_path=text_encoder_path,
        text_tokenizer=text_tokenizer_fn,
        image_preprocessor=image_preprocessor_fn,
        use_cpu_fallback=use_cpu_fallback
    )
    
    return app


def main():
    """Example usage of CLIP with QNN"""
    import os
    
    # Start timing the entire execution
    main_start_time = time.time()
    
    # Paths - use raw strings or forward slashes
    ONNX_MODEL_PATH = "models/clip_model.onnx"
    
    # Alternative: separate encoder paths if you have them
    # IMAGE_ENCODER_PATH = "models/clip_image_encoder.onnx"
    # TEXT_ENCODER_PATH = "models/clip_text_encoder.onnx"
    
    # Check if model exists
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: Model not found at {ONNX_MODEL_PATH}")
        return
    
    print("Initializing CLIP model...")
    init_start_time = time.time()
    
    # Initialize with combined model
    clip_app = initialize_clip_onnx(
        model_path=ONNX_MODEL_PATH,
        use_cpu_fallback=False  # Fall back to CPU if QNN fails
    )
    
    # Or initialize with separate encoders
    # clip_app = initialize_clip_onnx(
    #     image_encoder_path=IMAGE_ENCODER_PATH,
    #     text_encoder_path=TEXT_ENCODER_PATH,
    #     use_cpu_fallback=True
    # )
    
    init_end_time = time.time()
    print(f"Model initialization time: {(init_end_time - init_start_time)*1000:.2f} ms")
    
    print("\nRunning inference...")
    
    # Example images
    images = ["data/dr.jpg", "data/hd.jpeg"]
    
    # Example texts
    texts = [
        "a photo of a t-shirt",
        "a photo of a shirt",
        "a photo of a blouse",
        "a photo of a sweater",
        "a photo of a hoodie",
        "a photo of a jacket",
        "a photo of a coat",
        "a photo of a dress",
        "a photo of a skirt",
        "a photo of a pair of pants",
        "a photo of a pair of jeans",
        "a photo of a pair of shorts",
        "a photo of a suit",
        "a photo of a blazer",
        "a photo of a vest",
        "a photo of a jumpsuit",
        "a photo of a romper",
        "a photo of a pair of leggings",
        "a photo of a pair of sweatpants",
        "a photo of a pair of overalls",
        "a photo of a scarf",
        "a photo of a hat",
        "a photo of a cap",
        "a photo of a pair of gloves",
        "a photo of a pair of socks",
        "a photo of a pair of shoes",
        "a photo of a pair of boots",
        "a photo of a pair of sandals",
        "a photo of a tie",
        "a photo of a belt",
        "a photo of a hoodie",
        "a photo of a black hoodie",
    ]
    
    try:
        # Check if images exist
        for img_path in images:
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                images = []
                break
        
        if images:
            similarities = clip_app.predict_similarity(images, texts)
            print(f"\nSimilarity scores shape: {similarities.shape}")
            print("\nSimilarity Matrix:")
            print("-" * 50)
            
            # Print results in a nice format
            for i, img_path in enumerate(images):
                print(f"\nImage: {img_path}")
                for j, text in enumerate(texts):
                    score = similarities[i, j] if similarities.ndim > 1 else similarities[j]
                    print(f"  {text}: {score:.2f}%")
        else:
            print("No valid images found. Please check the data directory.")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    # Print total execution time
    main_end_time = time.time()
    print(f"\n" + "="*60)
    print(f"TOTAL EXECUTION TIME: {(main_end_time - main_start_time)*1000:.2f} ms")
    print(f"TOTAL EXECUTION TIME: {(main_end_time - main_start_time):.2f} seconds")
    print("="*60)


if __name__ == "__main__":
    main()