import os
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import logging
from huggingface_hub import snapshot_download

logging.set_verbosity_error()


class SettingsManager:
    DEFAULT = {
        "mode": "img2img",
        "model": "",
        "prompt": "2D cartoon illustration, bold black outlines, flat colors, clean linework, cell-shaded style, simple background, comic style, high detail, vibrant colors",
        "negative_prompt": "3D shadows, realistic lighting, gradients, textures, photorealism, blur, low quality, distortion, unwanted noise, overexposure, underexposure, bad anatomy",
        "content_prompt": "",  # new: separate prompt for text2img
        "img2img": {
            "strength": 0.4,
            "steps": 30,
            "guidance": 8.0,
            "resolution": "768x768",
            "size_option": "720p (1280x720)"
        },
        "text2img": {
            "width": 768,
            "height": 768,
            "steps": 30,
            "guidance": 7.5,
            "seed": -1
        },
        "upscaler": {
            "type": "Real-ESRGAN",
            "final_resolution": "1080p (1920x1080)",
            "sharpness": 1.0,
            "custom_width": 1920,    # custom upscale dimensions
            "custom_height": 1080
        },
        "vram_saver": False,
        "custom_models": []
    }

    def __init__(self, filename="setup.json"):
        self.filename = filename
        self.data = self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self.DEFAULT.copy()

    def save(self):
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()


# ----------------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------------
class Img2ImgApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Generator & Stylizer")
        self.root.geometry("900x800")
        self.root.minsize(800, 700)

        # Settings
        self.settings = SettingsManager()

        # Pipeline state
        self.pipe = None          # either StableDiffusionPipeline or StableDiffusionImg2ImgPipeline
        self.current_mode = None
        self.current_model_path = None
        self.model_loaded = False
        self.device = None

        # Image data
        self.input_image_path = None
        self.input_image_pil = None
        self.output_image = None
        self.upscaler = None

        # Build UI
        self.create_widgets()

        # Load saved settings into UI
        self.load_settings_into_ui()

    # ------------------------------------------------------------------
    # UI Creation
    # ------------------------------------------------------------------
    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ----- Tab 1: Generate -----
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Generate")

        # Mode selection
        mode_frame = ttk.LabelFrame(self.main_tab, text="Mode", padding=5)
        mode_frame.pack(fill=tk.X, pady=5)

        self.mode_var = tk.StringVar(value=self.settings.get("mode", "img2img"))
        ttk.Radiobutton(mode_frame, text="Image to Image", variable=self.mode_var,
                        value="img2img", command=self.on_mode_change).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Text to Image", variable=self.mode_var,
                        value="text2img", command=self.on_mode_change).pack(side=tk.LEFT, padx=10)

        # Model selection
        model_frame = ttk.LabelFrame(self.main_tab, text="Model", padding=5)
        model_frame.pack(fill=tk.X, pady=5)
        self.model_frame = model_frame

        top_row = ttk.Frame(model_frame)
        top_row.pack(fill=tk.X, pady=2)

        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(top_row, textvariable=self.model_var, width=50)
        model_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.model_combo = model_combo

        ttk.Button(top_row, text="Refresh", command=self.refresh_model_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_row, text="Add Folder...", command=self.add_model).pack(side=tk.LEFT, padx=2)
        self.load_model_btn = ttk.Button(top_row, text="Load Model", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=2)
        self.unload_model_btn = ttk.Button(top_row, text="Unload", command=self.unload_model, state=tk.DISABLED)
        self.unload_model_btn.pack(side=tk.LEFT, padx=2)

        # Download row
        download_frame = ttk.Frame(model_frame)
        download_frame.pack(fill=tk.X, pady=5)

        ttk.Label(download_frame, text="Download from Hugging Face:").pack(side=tk.LEFT, padx=5)
        self.model_id_var = tk.StringVar()
        ttk.Entry(download_frame, textvariable=self.model_id_var, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.download_btn = ttk.Button(download_frame, text="Download", command=self.download_model_thread)
        self.download_btn.pack(side=tk.LEFT, padx=5)

        # Mode-specific parameter frames
        self.img2img_frame = ttk.LabelFrame(self.main_tab, text="Image-to-Image Parameters", padding=5)
        self.text2img_frame = ttk.LabelFrame(self.main_tab, text="Text-to-Image Parameters", padding=5)

        # --- Img2Img controls ---
        # Strength
        ttk.Label(self.img2img_frame, text="Strength:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.strength_var = tk.DoubleVar(value=self.settings.get("img2img", {}).get("strength", 0.4))
        ttk.Scale(self.img2img_frame, from_=0.0, to=1.0, variable=self.strength_var,
                  orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5)
        self.strength_label = ttk.Label(self.img2img_frame, text=f"{self.strength_var.get():.2f}")
        self.strength_label.grid(row=0, column=2, padx=5)
        self.strength_var.trace("w", lambda *a: self.strength_label.config(text=f"{self.strength_var.get():.2f}"))

        # Steps
        ttk.Label(self.img2img_frame, text="Steps:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.steps_i2i_var = tk.IntVar(value=self.settings.get("img2img", {}).get("steps", 30))
        ttk.Scale(self.img2img_frame, from_=10, to=50, variable=self.steps_i2i_var,
                  orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5)
        self.steps_i2i_label = ttk.Label(self.img2img_frame, text=str(self.steps_i2i_var.get()))
        self.steps_i2i_label.grid(row=1, column=2, padx=5)
        self.steps_i2i_var.trace("w", lambda *a: self.steps_i2i_label.config(text=str(self.steps_i2i_var.get())))

        # Guidance
        ttk.Label(self.img2img_frame, text="Guidance:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.guidance_i2i_var = tk.DoubleVar(value=self.settings.get("img2img", {}).get("guidance", 8.0))
        ttk.Scale(self.img2img_frame, from_=1.0, to=15.0, variable=self.guidance_i2i_var,
                  orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, padx=5)
        self.guidance_i2i_label = ttk.Label(self.img2img_frame, text=f"{self.guidance_i2i_var.get():.1f}")
        self.guidance_i2i_label.grid(row=2, column=2, padx=5)
        self.guidance_i2i_var.trace("w", lambda *a: self.guidance_i2i_label.config(text=f"{self.guidance_i2i_var.get():.1f}"))

        # Gen resolution
        ttk.Label(self.img2img_frame, text="Gen Resolution:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.res_i2i_var = tk.StringVar(value=self.settings.get("img2img", {}).get("resolution", "768x768"))
        ttk.Combobox(self.img2img_frame, textvariable=self.res_i2i_var,
                     values=["512x512", "768x768", "1024x1024"], width=15).grid(row=3, column=1, sticky=tk.W, padx=5)

        # Output size (no upscaler)
        ttk.Label(self.img2img_frame, text="Output Size (no upscaler):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.size_i2i_var = tk.StringVar(value=self.settings.get("img2img", {}).get("size_option", "720p (1280x720)"))
        ttk.Combobox(self.img2img_frame, textvariable=self.size_i2i_var,
                     values=["720p (1280x720)", "Original proportions"], width=20).grid(row=4, column=1, sticky=tk.W, padx=5)

        # --- Text2Img controls ---
        ttk.Label(self.text2img_frame, text="Width:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.width_var = tk.IntVar(value=self.settings.get("text2img", {}).get("width", 768))
        ttk.Spinbox(self.text2img_frame, from_=256, to=1024, increment=64,
                    textvariable=self.width_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(self.text2img_frame, text="Height:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.height_var = tk.IntVar(value=self.settings.get("text2img", {}).get("height", 768))
        ttk.Spinbox(self.text2img_frame, from_=256, to=1024, increment=64,
                    textvariable=self.height_var, width=8).grid(row=1, column=1, padx=5)

        ttk.Label(self.text2img_frame, text="Steps:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.steps_t2i_var = tk.IntVar(value=self.settings.get("text2img", {}).get("steps", 30))
        ttk.Scale(self.text2img_frame, from_=10, to=50, variable=self.steps_t2i_var,
                  orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, padx=5)
        self.steps_t2i_label = ttk.Label(self.text2img_frame, text=str(self.steps_t2i_var.get()))
        self.steps_t2i_label.grid(row=2, column=2, padx=5)
        self.steps_t2i_var.trace("w", lambda *a: self.steps_t2i_label.config(text=str(self.steps_t2i_var.get())))

        ttk.Label(self.text2img_frame, text="Guidance:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.guidance_t2i_var = tk.DoubleVar(value=self.settings.get("text2img", {}).get("guidance", 7.5))
        ttk.Scale(self.text2img_frame, from_=1.0, to=15.0, variable=self.guidance_t2i_var,
                  orient=tk.HORIZONTAL, length=200).grid(row=3, column=1, padx=5)
        self.guidance_t2i_label = ttk.Label(self.text2img_frame, text=f"{self.guidance_t2i_var.get():.1f}")
        self.guidance_t2i_label.grid(row=3, column=2, padx=5)
        self.guidance_t2i_var.trace("w", lambda *a: self.guidance_t2i_label.config(text=f"{self.guidance_t2i_var.get():.1f}"))

        ttk.Label(self.text2img_frame, text="Seed (-1=random):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.seed_var = tk.IntVar(value=self.settings.get("text2img", {}).get("seed", -1))
        ttk.Spinbox(self.text2img_frame, from_=-1, to=2**32-1, increment=1,
                    textvariable=self.seed_var, width=12).grid(row=4, column=1, sticky=tk.W, padx=5)

        # --- Prompts ---
        prompt_frame = ttk.LabelFrame(self.main_tab, text="Prompts", padding=5)
        prompt_frame.pack(fill=tk.X, pady=5)

        # Style prompt (used for both modes, but for text2img it's the main prompt)
        ttk.Label(prompt_frame, text="Style Prompt:").pack(anchor=tk.W)
        self.prompt_text = tk.Text(prompt_frame, height=3, width=80)
        self.prompt_text.pack(fill=tk.X, pady=2)

        # Content prompt (only visible in text2img mode, overrides style prompt if non-empty)
        self.content_prompt_frame = ttk.Frame(prompt_frame)
        ttk.Label(self.content_prompt_frame, text="Content Prompt (text2img only, overrides style):").pack(anchor=tk.W)
        self.content_prompt_text = tk.Text(self.content_prompt_frame, height=2, width=80)
        self.content_prompt_text.pack(fill=tk.X, pady=2)

        # Negative prompt
        ttk.Label(prompt_frame, text="Negative Prompt:").pack(anchor=tk.W)
        self.neg_prompt_text = tk.Text(prompt_frame, height=2, width=80)
        self.neg_prompt_text.pack(fill=tk.X, pady=2)

        # --- Upscaling & Enhancement ---
        upscale_frame = ttk.LabelFrame(self.main_tab, text="Upscaling & Enhancement", padding=5)
        upscale_frame.pack(fill=tk.X, pady=5)

        ttk.Label(upscale_frame, text="Upscaler:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.upscaler_var = tk.StringVar(value=self.settings.get("upscaler", {}).get("type", "Real-ESRGAN"))
        upscaler_combo = ttk.Combobox(upscale_frame, textvariable=self.upscaler_var,
                                      values=["None (Resize only)", "Real-ESRGAN", "Stable Diffusion img2img (slow)"],
                                      width=25)
        upscaler_combo.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(upscale_frame, text="Final Resolution:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.out_res_var = tk.StringVar(value=self.settings.get("upscaler", {}).get("final_resolution", "1080p (1920x1080)"))
        out_res_combo = ttk.Combobox(upscale_frame, textvariable=self.out_res_var,
                                     values=["1080p (1920x1080)", "1440p (2560x1440)", "4K (3840x2160)", "Custom"],
                                     width=25)
        out_res_combo.grid(row=1, column=1, sticky=tk.W, padx=5)

        # Custom resolution frame (initially hidden)
        self.custom_res_frame = ttk.Frame(upscale_frame)
        ttk.Label(self.custom_res_frame, text="Width:").pack(side=tk.LEFT, padx=5)
        self.custom_width_var = tk.IntVar(value=self.settings.get("upscaler", {}).get("custom_width", 1920))
        ttk.Spinbox(self.custom_res_frame, from_=1, to=8192, increment=1,
                    textvariable=self.custom_width_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.custom_res_frame, text="Height:").pack(side=tk.LEFT, padx=5)
        self.custom_height_var = tk.IntVar(value=self.settings.get("upscaler", {}).get("custom_height", 1080))
        ttk.Spinbox(self.custom_res_frame, from_=1, to=8192, increment=1,
                    textvariable=self.custom_height_var, width=8).pack(side=tk.LEFT, padx=5)

        self.custom_res_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        self.custom_res_frame.grid_remove()  # hide initially

        def on_res_change(*args):
            if self.out_res_var.get() == "Custom":
                self.custom_res_frame.grid()
            else:
                self.custom_res_frame.grid_remove()
            self.save_upscaler_param("final_resolution", self.out_res_var.get())
        self.out_res_var.trace("w", on_res_change)

        # Sharpness
        ttk.Label(upscale_frame, text="Sharpness:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.sharpness_var = tk.DoubleVar(value=self.settings.get("upscaler", {}).get("sharpness", 1.0))
        sharp_scale = ttk.Scale(upscale_frame, from_=0.0, to=2.0, variable=self.sharpness_var,
                                orient=tk.HORIZONTAL, length=150)
        sharp_scale.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.sharpness_label = ttk.Label(upscale_frame, text=f"{self.sharpness_var.get():.1f}")
        self.sharpness_label.grid(row=3, column=2, padx=5)
        self.sharpness_var.trace("w", lambda *a: self.sharpness_label.config(text=f"{self.sharpness_var.get():.1f}"))

        # VRAM Saver
        self.vram_saver_var = tk.BooleanVar(value=self.settings.get("vram_saver", False))
        ttk.Checkbutton(upscale_frame, text="VRAM Saver Mode (slower, uses less VRAM)",
                        variable=self.vram_saver_var).grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

        # Generate button
        self.generate_btn = ttk.Button(self.main_tab, text="Generate", command=self.generate_thread)
        self.generate_btn.pack(pady=10)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.main_tab, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=5)

        # ----- Tab 2: Preview -----
        self.preview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_tab, text="Preview")

        preview_frame = ttk.Frame(self.preview_tab)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        input_frame = ttk.LabelFrame(preview_frame, text="Input Image")
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.input_canvas = tk.Canvas(input_frame, width=300, height=300, bg='gray')
        self.input_canvas.pack(padx=5, pady=5)

        output_frame = ttk.LabelFrame(preview_frame, text="Output Image")
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.output_canvas = tk.Canvas(output_frame, width=300, height=300, bg='gray')
        self.output_canvas.pack(padx=5, pady=5)

        btn_frame = ttk.Frame(self.preview_tab)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Select Input Image", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Output Image", command=self.save_output).pack(side=tk.LEFT, padx=5)

        self.bind_settings_save()

    # ------------------------------------------------------------------
    # UI Helpers
    # ------------------------------------------------------------------
    def on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "img2img":
            self.text2img_frame.pack_forget()
            self.img2img_frame.pack(fill=tk.X, pady=5)
            self.content_prompt_frame.pack_forget()
        else:
            self.img2img_frame.pack_forget()
            self.text2img_frame.pack(fill=tk.X, pady=5)
        # Pack the content prompt frame right after the style prompt text
            self.content_prompt_frame.pack(fill=tk.X, pady=2, after=self.prompt_text)

    # Reload model if necessary (already implemented)
        if self.model_loaded and self.current_model_path:
            self.unload_model()
            self.load_model(force=True)

        self.settings.set("mode", mode)

    def load_settings_into_ui(self):
        mode = self.settings.get("mode", "img2img")
        self.mode_var.set(mode)
        self.on_mode_change()  # shows correct frames

        # Prompts
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(1.0, self.settings.get("prompt", ""))
        self.content_prompt_text.delete(1.0, tk.END)
        self.content_prompt_text.insert(1.0, self.settings.get("content_prompt", ""))
        self.neg_prompt_text.delete(1.0, tk.END)
        self.neg_prompt_text.insert(1.0, self.settings.get("negative_prompt", ""))

        # Img2Img
        i2i = self.settings.get("img2img", {})
        self.strength_var.set(i2i.get("strength", 0.4))
        self.steps_i2i_var.set(i2i.get("steps", 30))
        self.guidance_i2i_var.set(i2i.get("guidance", 8.0))
        self.res_i2i_var.set(i2i.get("resolution", "768x768"))
        self.size_i2i_var.set(i2i.get("size_option", "720p (1280x720)"))

        # Text2Img
        t2i = self.settings.get("text2img", {})
        self.width_var.set(t2i.get("width", 768))
        self.height_var.set(t2i.get("height", 768))
        self.steps_t2i_var.set(t2i.get("steps", 30))
        self.guidance_t2i_var.set(t2i.get("guidance", 7.5))
        self.seed_var.set(t2i.get("seed", -1))

        # Upscaler
        up = self.settings.get("upscaler", {})
        self.upscaler_var.set(up.get("type", "Real-ESRGAN"))
        self.out_res_var.set(up.get("final_resolution", "1080p (1920x1080)"))
        self.sharpness_var.set(up.get("sharpness", 1.0))
        self.custom_width_var.set(up.get("custom_width", 1920))
        self.custom_height_var.set(up.get("custom_height", 1080))
        self.vram_saver_var.set(self.settings.get("vram_saver", False))

        # Model list
        self.refresh_model_list()
        model_path = self.settings.get("model", "")
        if model_path:
            self.model_var.set(model_path)

    def bind_settings_save(self):
        """Bind changes in UI variables to auto-save settings."""
        # Prompts
        def save_prompts(*args):
            self.settings.set("prompt", self.prompt_text.get(1.0, tk.END).strip())
            self.settings.set("negative_prompt", self.neg_prompt_text.get(1.0, tk.END).strip())
            self.settings.set("content_prompt", self.content_prompt_text.get(1.0, tk.END).strip())
        self.prompt_text.bind("<KeyRelease>", lambda e: save_prompts())
        self.neg_prompt_text.bind("<KeyRelease>", lambda e: save_prompts())
        self.content_prompt_text.bind("<KeyRelease>", lambda e: save_prompts())
        # Mode
        self.mode_var.trace("w", lambda *a: self.settings.set("mode", self.mode_var.get()))

        # Img2Img
        self.strength_var.trace("w", lambda *a: self.save_img2img_param("strength", self.strength_var.get()))
        self.steps_i2i_var.trace("w", lambda *a: self.save_img2img_param("steps", self.steps_i2i_var.get()))
        self.guidance_i2i_var.trace("w", lambda *a: self.save_img2img_param("guidance", self.guidance_i2i_var.get()))
        self.res_i2i_var.trace("w", lambda *a: self.save_img2img_param("resolution", self.res_i2i_var.get()))
        self.size_i2i_var.trace("w", lambda *a: self.save_img2img_param("size_option", self.size_i2i_var.get()))

        # Text2Img
        self.width_var.trace("w", lambda *a: self.save_text2img_param("width", self.width_var.get()))
        self.height_var.trace("w", lambda *a: self.save_text2img_param("height", self.height_var.get()))
        self.steps_t2i_var.trace("w", lambda *a: self.save_text2img_param("steps", self.steps_t2i_var.get()))
        self.guidance_t2i_var.trace("w", lambda *a: self.save_text2img_param("guidance", self.guidance_t2i_var.get()))
        self.seed_var.trace("w", lambda *a: self.save_text2img_param("seed", self.seed_var.get()))

        # Upscaler
        self.upscaler_var.trace("w", lambda *a: self.save_upscaler_param("type", self.upscaler_var.get()))
        self.out_res_var.trace("w", lambda *a: self.save_upscaler_param("final_resolution", self.out_res_var.get()))
        self.sharpness_var.trace("w", lambda *a: self.save_upscaler_param("sharpness", self.sharpness_var.get()))
        self.vram_saver_var.trace("w", lambda *a: self.settings.set("vram_saver", self.vram_saver_var.get()))
# Add traces for custom width/height
        self.custom_width_var.trace("w", lambda *a: self.save_upscaler_param("custom_width", self.custom_width_var.get()))
        self.custom_height_var.trace("w", lambda *a: self.save_upscaler_param("custom_height", self.custom_height_var.get()))

        # Model
        def on_model_changed(*args):
            self.settings.set("model", self.model_var.get())
        self.model_var.trace("w", on_model_changed)

    def save_img2img_param(self, key, value):
        i2i = self.settings.get("img2img", {})
        i2i[key] = value
        self.settings.set("img2img", i2i)

    def save_text2img_param(self, key, value):
        t2i = self.settings.get("text2img", {})
        t2i[key] = value
        self.settings.set("text2img", t2i)

    def save_upscaler_param(self, key, value):
        up = self.settings.get("upscaler", {})
        up[key] = value
        self.settings.set("upscaler", up)

    def clear_memory_cache(self):
        """Clear PyTorch memory cache for CUDA or DirectML."""
        if hasattr(torch, 'directml') and hasattr(torch.directml, 'empty_cache'):
            torch.directml.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # ------------------------------------------------------------------
    # Model Management
    # ------------------------------------------------------------------
    def refresh_model_list(self):
        """Scan the 'models' folder and combine with saved custom paths."""
        models = set()
        # Look for subfolders in a "models" directory
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if os.path.isdir(models_dir):
            for item in os.listdir(models_dir):
                full = os.path.join(models_dir, item)
                if os.path.isdir(full):
                    models.add(full)

        # Add saved custom model paths
        custom = self.settings.get("custom_models", [])
        for path in custom:
            if os.path.exists(path):
                models.add(path)

        # Convert to list for combobox
        model_list = sorted(models)
        self.model_combo["values"] = model_list
        if not model_list:
            self.model_combo.set("")
            return

        # If current model is in list, keep it, else select first
        current = self.model_var.get()
        if current in model_list:
            self.model_var.set(current)
        else:
            self.model_var.set(model_list[0])

    def add_model(self):
        """Open folder dialog to add a new model (containing a 'model_index.json' file)."""
        folder = filedialog.askdirectory(title="Select a model folder (containing model_index.json)")
        if folder:
            custom = self.settings.get("custom_models", [])
            if folder not in custom:
                custom.append(folder)
                self.settings.set("custom_models", custom)
                self.refresh_model_list()
# --- Updated load_model with optional force ---
    def load_model(self, force=False):
        model_path = self.model_var.get()
        if not model_path:
            messagebox.showwarning("No Model", "Please select a model first.")
            return
        if self.model_loaded and self.current_model_path == model_path and self.current_mode == self.mode_var.get() and not force:
            self.status_var.set("Model already loaded.")
            return

        # Unload previous
        self.unload_model()

        self.status_var.set(f"Loading model from {model_path}...")
        self.load_model_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.DISABLED)

        def load():
            try:
                try:
                    import torch_directml
                    self.device = torch_directml.device()
                except ImportError:
                    self.device = torch.device("cpu")

                mode = self.mode_var.get()
                if mode == "img2img":
                    pipe_class = StableDiffusionImg2ImgPipeline
                else:
                    pipe_class = StableDiffusionPipeline

                kwargs = {
                    "pretrained_model_name_or_path": model_path,
                    "torch_dtype": torch.float16,
                    "safety_checker": None,
                    "requires_safety_checker": False,
                }
                self.pipe = pipe_class.from_pretrained(**kwargs)
                self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

                self.pipe.enable_attention_slicing()
                if hasattr(self.pipe, 'enable_vae_slicing'):
                    self.pipe.enable_vae_slicing()

                if self.vram_saver_var.get():
                    self.pipe.enable_model_cpu_offload()
                    self.device = None
                else:
                    self.pipe = self.pipe.to(self.device)

                self.current_model_path = model_path
                self.current_mode = mode
                self.model_loaded = True
                self.status_var.set("Model loaded successfully.")
                self.generate_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.status_var.set(f"Failed to load model: {e}")
                messagebox.showerror("Model Load Error", str(e))
            finally:
                self.load_model_btn.config(state=tk.NORMAL)
                self.unload_model_btn.config(state=tk.NORMAL if self.model_loaded else tk.DISABLED)

        threading.Thread(target=load, daemon=True).start()

    def unload_model(self):
        """Free the pipeline and release memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.model_loaded = False
        self.current_model_path = None
        self.current_mode = None
        self.unload_model_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.DISABLED if not self.model_loaded else tk.NORMAL)
        self.status_var.set("Model unloaded.")
        self.clear_memory_cache()

# ------------------------------------------------------------------
    # Model Download
    # ------------------------------------------------------------------
    def download_model_thread(self):
        """Start a thread to download a model."""
        model_id = self.model_id_var.get().strip()
        if not model_id:
            messagebox.showwarning("No model ID", "Please enter a Hugging Face model ID (e.g., Lykon/dreamshaper-8).")
            return

        # Disable download button during download
        self.download_btn.config(state=tk.DISABLED)
        self.status_var.set(f"Downloading {model_id}...")

        def download():
            try:
                # Create a safe folder name from model_id (replace / with _)
                safe_name = model_id.replace('/', '_')
                models_dir = os.path.join(os.path.dirname(__file__), "models")
                os.makedirs(models_dir, exist_ok=True)
                target_folder = os.path.join(models_dir, safe_name)

                # Check if already exists
                if os.path.exists(target_folder):
                    self.status_var.set(f"Model already exists at {target_folder}")
                    # Still add to custom_models if not already there
                    custom = self.settings.get("custom_models", [])
                    if target_folder not in custom:
                        custom.append(target_folder)
                        self.settings.set("custom_models", custom)
                        self.refresh_model_list()
                    self.download_btn.config(state=tk.NORMAL)
                    return

                # Download using snapshot_download
                snapshot_download(
                    repo_id=model_id,
                    local_dir=target_folder,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

                # Add to custom models list and refresh
                custom = self.settings.get("custom_models", [])
                if target_folder not in custom:
                    custom.append(target_folder)
                    self.settings.set("custom_models", custom)
                self.refresh_model_list()
                self.status_var.set(f"Downloaded {model_id} to {target_folder}")
                messagebox.showinfo("Download Complete", f"Model {model_id} has been downloaded.")
            except Exception as e:
                self.status_var.set(f"Download failed: {e}")
                messagebox.showerror("Download Error", f"Failed to download {model_id}\n{str(e)}")
            finally:
                self.download_btn.config(state=tk.NORMAL)

        threading.Thread(target=download, daemon=True).start()


    # ------------------------------------------------------------------
    # Image Handling
    # ------------------------------------------------------------------
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            self.input_image_path = file_path
            self.input_image_pil = Image.open(file_path).convert("RGB")
            # Display preview
            disp_img = self.input_image_pil.copy()
            disp_img.thumbnail((300, 300))
            self.input_photo = ImageTk.PhotoImage(disp_img)
            self.input_canvas.delete("all")
            self.input_canvas.create_image(150, 150, image=self.input_photo, anchor=tk.CENTER)
            self.input_canvas.config(width=300, height=300)

    def save_output(self):
        if self.output_image is None:
            messagebox.showinfo("No Output", "No output image to save.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            title="Save stylized image as"
        )
        if save_path:
            self.output_image.save(save_path)
            self.status_var.set(f"Saved to {save_path}")

    def load_upscaler(self):
        """Lazily load Real-ESRGAN upscaler."""
        if hasattr(self, 'upscaler') and self.upscaler is not None:
            return True
        try:
            from realesrgan import RealESRGANer
            # If DirectML is in use, force CPU (Real-ESRGAN may not support DirectML)
            if hasattr(torch, 'directml'):
                device = torch.device('cpu')
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.upscaler = RealESRGANer(
                scale=4,
                model_path=None,
                device=device,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False
            )
            self.status_var.set("Real-ESRGAN upscaler loaded.")
            return True
        except ImportError:
            self.status_var.set("Real-ESRGAN not installed. Using simple resize.")
            self.upscaler = None
            return False
        except Exception as e:
            self.status_var.set(f"Failed to load upscaler: {e}")
            self.upscaler = None
            return False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Image Generation with Real-Time Preview & VRAM Cleanup
    # ------------------------------------------------------------------
    def generate_thread(self):
        if not self.model_loaded:
            messagebox.showwarning("Model not loaded", "Please load a model first using the 'Load Model' button.")
            return

        mode = self.mode_var.get()
        if mode == "img2img" and not self.input_image_path:
            messagebox.showwarning("No image", "Please select an input image for img2img.")
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.status_var.set("Generating...")

        def generate():
            try:
                style_prompt = self.prompt_text.get("1.0", tk.END).strip()
                neg_prompt = self.neg_prompt_text.get("1.0", tk.END).strip()
                content_prompt = self.content_prompt_text.get("1.0", tk.END).strip()

                if mode == "text2img" and content_prompt:
                    prompt = content_prompt
                else:
                    prompt = style_prompt

                # ----- Helper to decode latents to a PIL image (with memory cleanup) -----
                def decode_latents(latents):
                    with torch.no_grad():
                        latents = 1 / 0.18215 * latents
                        latents = latents.to(self.pipe.vae.device)
                        image = self.pipe.vae.decode(latents).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.cpu().permute(0, 2, 3, 1).numpy()
                        # Delete the large tensor to free memory earlier
                        del latents
                        return Image.fromarray((image[0] * 255).astype('uint8'))

                # ----- Preview update (runs in main thread) -----
                def update_preview(latents):
                    try:
                        if self.pipe is None or not self.model_loaded:
                            return
                        img = decode_latents(latents)
                        out_disp = img.copy()
                        out_disp.thumbnail((300, 300))
                        self.output_photo = ImageTk.PhotoImage(out_disp)
                        self.output_canvas.delete("all")
                        self.output_canvas.create_image(150, 150, image=self.output_photo, anchor=tk.CENTER)
                        self.output_canvas.config(width=300, height=300)
                    except Exception as e:
                        print(f"Preview error: {e}")

                # ----- Callback for intermediate steps (update less often to save VRAM) -----
                def step_callback(step, timestep, latents):
                    if step % 4 == 0:   # reduced from 2 to 4 to reduce VAE decodes
                        self.root.after(0, lambda: update_preview(latents))

                if mode == "img2img":
                    strength = self.strength_var.get()
                    steps = self.steps_i2i_var.get()
                    guidance = self.guidance_i2i_var.get()
                    res_str = self.res_i2i_var.get()
                    if res_str == "512x512":
                        gen_size = (512, 512)
                    elif res_str == "768x768":
                        gen_size = (768, 768)
                    else:
                        gen_size = (1024, 1024)

                    init_image = self.input_image_pil.resize(gen_size, Image.Resampling.LANCZOS)

                    with torch.no_grad():
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=neg_prompt,
                            image=init_image,
                            strength=strength,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            callback=step_callback,
                            callback_steps=1,
                        ).images[0]
                else:  # text2img
                    width = self.width_var.get()
                    height = self.height_var.get()
                    steps = self.steps_t2i_var.get()
                    guidance = self.guidance_t2i_var.get()
                    seed = self.seed_var.get()
                    if seed == -1:
                        seed = None

                    generator = torch.Generator(device=self.device or "cpu")
                    if seed is not None:
                        generator = generator.manual_seed(seed)

                    with torch.no_grad():
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=neg_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            generator=generator,
                            callback=step_callback,
                            callback_steps=1,
                        ).images[0]

                # --- Upscaling ---
                upscaler_type = self.upscaler_var.get()
                out_res = self.out_res_var.get()
                if out_res == "Custom":
                    target_size = (self.custom_width_var.get(), self.custom_height_var.get())
                elif out_res == "1080p (1920x1080)":
                    target_size = (1920, 1080)
                elif out_res == "1440p (2560x1440)":
                    target_size = (2560, 1440)
                elif out_res == "4K (3840x2160)":
                    target_size = (3840, 2160)
                else:
                    target_size = None

                if upscaler_type == "Real-ESRGAN" and target_size is not None:
                    self.status_var.set("Upscaling with Real-ESRGAN...")
                    if self.load_upscaler() and self.upscaler is not None:
                        import numpy as np
                        img_np = np.array(result)
                        output, _ = self.upscaler.enhance(img_np, outscale=4)
                        result = Image.fromarray(output)
                        if result.size != target_size:
                            result = result.resize(target_size, Image.Resampling.LANCZOS)
                    else:
                        result = result.resize(target_size, Image.Resampling.LANCZOS)
                elif upscaler_type == "None (Resize only)" and target_size is not None:
                    result = result.resize(target_size, Image.Resampling.LANCZOS)
                elif mode == "img2img" and upscaler_type != "None (Resize only)":
                    size_option = self.size_i2i_var.get()
                    if size_option.startswith("720p"):
                        result = result.resize((1280, 720), Image.Resampling.LANCZOS)

                # Sharpness
                sharpness = self.sharpness_var.get()
                if sharpness != 1.0:
                    enhancer = ImageEnhance.Sharpness(result)
                    result = enhancer.enhance(sharpness)

                self.output_image = result

                # Final preview update
                out_disp = result.copy()
                out_disp.thumbnail((300, 300))
                self.output_photo = ImageTk.PhotoImage(out_disp)
                self.output_canvas.delete("all")
                self.output_canvas.create_image(150, 150, image=self.output_photo, anchor=tk.CENTER)
                self.output_canvas.config(width=300, height=300)

                self.status_var.set("Generation complete.")
                if messagebox.askyesno("Save Output", "Do you want to save the generated image?"):
                    self.save_output()

            except Exception as e:
                self.status_var.set(f"Error: {e}")
                messagebox.showerror("Error", str(e))
            finally:
                self.generate_btn.config(state=tk.NORMAL)
                # Use the safe memory cleanup method
                self.clear_memory_cache()
                # Also clear any leftover references
                if 'result' in locals():
                    del result

        threading.Thread(target=generate, daemon=True).start()

# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = Img2ImgApp(root)
    root.mainloop()