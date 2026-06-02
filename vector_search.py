import os

os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import SiglipProcessor, SiglipModel, BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
from pathlib import Path

class VectorSearchService:
    _instance = None

    @staticmethod
    def _ensure_safetensors_snapshot(model_id: str) -> None:
        """
        Hugging Face cache snapshots might have model.safetensors in one snapshot
        and config.json/vocab.txt in another snapshot. To avoid the torch.load
        vulnerability error under older torch versions, we dynamically copy/link
        the metadata files to the safetensors snapshot so it can be loaded fully using safetensors.
        """
        log_file = Path("sync_debug.log")
        try:
            with open(log_file, "a") as log:
                log.write(f"\n--- SYNC START for {model_id} ---\n")
                cache_name = f"models--{model_id.replace('/', '--')}"
                search_dirs = [
                    Path("hf_cache") / "hub" / cache_name / "snapshots",
                    Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
                ]
                for snapshots_dir in search_dirs:
                    log.write(f"Checking snapshots_dir: {snapshots_dir} (exists={snapshots_dir.exists()})\n")
                    if not snapshots_dir.exists():
                        continue
                    
                    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
                    log.write(f"Found snapshots: {[s.name for s in snapshots]}\n")
                    if not snapshots:
                        continue
                        
                    safetensors_snapshot = None
                    metadata_snapshot = None
                    
                    for snapshot in snapshots:
                        has_safetensors = (snapshot / "model.safetensors").exists()
                        has_config = (snapshot / "config.json").exists()
                        has_vocab = (snapshot / "vocab.txt").exists()
                        log.write(f"Snapshot: {snapshot.name} | has_safetensors={has_safetensors} | has_config={has_config} | has_vocab={has_vocab}\n")
                        
                        if has_safetensors:
                            safetensors_snapshot = snapshot
                        if has_config and has_vocab:
                            metadata_snapshot = snapshot
                            
                    log.write(f"safetensors_snapshot: {safetensors_snapshot.name if safetensors_snapshot else 'None'}\n")
                    log.write(f"metadata_snapshot: {metadata_snapshot.name if metadata_snapshot else 'None'}\n")
                    
                    if safetensors_snapshot and metadata_snapshot and safetensors_snapshot != metadata_snapshot:
                        files_to_copy = [
                            "config.json", 
                            "preprocessor_config.json", 
                            "tokenizer.json", 
                            "tokenizer_config.json", 
                            "vocab.txt", 
                            "special_tokens_map.json"
                        ]
                        import shutil
                        for filename in files_to_copy:
                            src = metadata_snapshot / filename
                            dst = safetensors_snapshot / filename
                            log.write(f"Checking file to copy: {filename} | src_exists={src.exists()} | dst_exists={dst.exists()}\n")
                            if src.exists() and not dst.exists():
                                try:
                                    shutil.copy2(src, dst)
                                    log.write(f"Copied {filename} to {safetensors_snapshot.name}\n")
                                    print(f"[BLIP Sync] Copied {filename} to safetensors snapshot.")
                                except Exception as e:
                                    log.write(f"Failed to copy {filename}: {e}\n")
                                    print(f"[BLIP Sync] Failed to copy {filename}: {e}")
        except Exception as e:
            print(f"[BLIP Sync] Sync execution error: {e}")

    @staticmethod
    def _local_model_path(model_id: str, required_files: tuple[str, ...]) -> str:
        cache_name = f"models--{model_id.replace('/', '--')}"
        # Search both the new E-drive cache (hf_cache) and standard C-drive cache
        search_dirs = [
            Path("hf_cache") / "hub" / cache_name / "snapshots",
            Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
        ]
        for snapshots_dir in search_dirs:
            if snapshots_dir.exists():
                snapshots = sorted(
                    [path for path in snapshots_dir.iterdir() if path.is_dir()],
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                for snapshot in snapshots:
                    if all((snapshot / filename).exists() for filename in required_files):
                        return str(snapshot)
        return model_id

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorSearchService, cls).__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading AI models on {cls._instance.device}...")
            
            # --- Google SigLIP (For Semantic Vector Search - 768d) ---
            siglip_id = "google/siglip-base-patch16-256"
            siglip_path = cls._local_model_path(
                siglip_id,
                ("config.json", "model.safetensors", "spiece.model"),
            )
            siglip_local_only = siglip_path != siglip_id
            cls._instance.siglip_processor = SiglipProcessor.from_pretrained(
                siglip_path,
                local_files_only=siglip_local_only,
            )
            cls._instance.siglip_model = SiglipModel.from_pretrained(
                siglip_path,
                local_files_only=siglip_local_only,
            ).to(cls._instance.device)
            cls._instance.siglip_model.eval()

            # --- Salesforce BLIP (For Image Captioning) ---
            blip_id = "Salesforce/blip-image-captioning-base"
            
            # Synchronize cache files to prefer secure model.safetensors loading
            try:
                cls._ensure_safetensors_snapshot(blip_id)
            except Exception as e:
                print(f"[BLIP Sync] Error processing cache sync: {e}")
                
            blip_path = cls._local_model_path(
                blip_id,
                ("config.json", "model.safetensors", "vocab.txt"),
            )
            if blip_path == blip_id:
                blip_path = cls._local_model_path(
                    blip_id,
                    ("config.json", "pytorch_model.bin", "vocab.txt"),
                )
            blip_local_only = blip_path != blip_id
            
            print(f"Loading BLIP from: {blip_path} (local_only={blip_local_only})")
            cls._instance.blip_processor = BlipProcessor.from_pretrained(
                blip_path,
                local_files_only=blip_local_only,
            )
            cls._instance.blip_model = BlipForConditionalGeneration.from_pretrained(
                blip_path,
                local_files_only=blip_local_only,
            ).to(cls._instance.device)
            cls._instance.blip_model.eval()
            
            print("AI models loaded successfully.")
        return cls._instance

    def generate_caption(self, image: Image.Image) -> str:
        try:
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_new_tokens=50)
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"[BLIP] Caption generation error: {e}")
            return ""

    def get_image_embedding(self, image: Image.Image) -> list:
        inputs = self.siglip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.siglip_model.get_image_features(**inputs)
            if hasattr(output, "pooler_output"):
                output = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state.mean(dim=1)
            # Normalize for cosine similarity
            features = output / output.norm(p=2, dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def get_text_embedding(self, text: str) -> list:
        inputs = self.siglip_processor(text=[text], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.siglip_model.get_text_features(**inputs)
            if hasattr(output, "pooler_output"):
                output = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state.mean(dim=1)
            # Normalize for cosine similarity
            features = output / output.norm(p=2, dim=-1, keepdim=True)
        return features[0].cpu().tolist()

# Global instance (singleton — only constructed once)
vector_service = VectorSearchService()

