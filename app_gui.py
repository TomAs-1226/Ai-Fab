import os
import json
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

########################################
# CONFIG
########################################

# Directory where we keep per-model memory JSON files
MEMORY_DIR = "./memory_store"
os.makedirs(MEMORY_DIR, exist_ok=True)

# How many previous turns we stuff back into the context
MAX_CONTEXT_MESSAGES = 10

# Generation defaults (can be tuned in UI)
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9

########################################
# HELPER: basic markdown -> tkinter tags
########################################

def apply_markdown_tags(text_widget: scrolledtext.ScrolledText, start_index: str, content: str):
    """
    Render lightweight markdown styles into a tk Text widget:
    - **bold**
    - *italic*
    - # Heading
    We'll do a simple pass:
      1) Insert raw text without markup
      2) Track ranges for bold/italic/heading
    We'll rebuild & insert segments. Simpler approach: parse->insert piece by piece.
    """

    # We'll do a tiny inline parser.
    # Rules:
    #   bold: **...**
    #   ital: *...*
    #   heading line: r"^# (.+)$"
    #
    # NOTE: This is not a full markdown parser, just enough for readable bold etc.
    import re

    # Split content by lines so we can detect heading lines
    lines = content.split("\n")

    cur_line_index = start_index

    # define tags upfront
    text_widget.tag_configure("bold", font=("Segoe UI", 10, "bold"))
    text_widget.tag_configure("italic", font=("Segoe UI", 10, "italic"))
    text_widget.tag_configure("heading", font=("Segoe UI Semibold", 11, "bold"))

    for i, raw_line in enumerate(lines):
        # heading?
        heading_match = re.match(r"^# (.+)$", raw_line.strip())
        if heading_match:
            # Insert heading line as a single chunk tagged "heading"
            line_text = heading_match.group(1) + "\n"
            start_pos = text_widget.index("end")
            text_widget.insert("end", line_text)
            end_pos = text_widget.index("end")
            text_widget.tag_add("heading", start_pos, end_pos)
        else:
            # parse bold / italic inline
            # We'll do nested-safe-ish approach:
            # Replace **bold** first
            segments = []
            tmp = raw_line

            bold_pattern = r"\*\*(.+?)\*\*"
            ital_pattern = r"\*(.+?)\*"

            # We'll walk the string and create (text, tag) segments
            cursor = 0
            for m in re.finditer(bold_pattern, tmp):
                if m.start() > cursor:
                    segments.append((tmp[cursor:m.start()], None))
                segments.append((m.group(1), "bold"))
                cursor = m.end()
            if cursor < len(tmp):
                segments.append((tmp[cursor:], None))

            # Now run italic inside each non-tagged None segment
            final_segments = []
            for seg_text, seg_tag in segments:
                if seg_tag is not None:
                    # already bold, push directly
                    final_segments.append((seg_text, seg_tag))
                else:
                    # break this segment for italics
                    cursor2 = 0
                    for m2 in re.finditer(ital_pattern, seg_text):
                        if m2.start() > cursor2:
                            final_segments.append((seg_text[cursor2:m2.start()], None))
                        final_segments.append((m2.group(1), "italic"))
                        cursor2 = m2.end()
                    if cursor2 < len(seg_text):
                        final_segments.append((seg_text[cursor2:], None))

            # Insert final segments into widget
            for seg_text, seg_tag in final_segments:
                start_pos = text_widget.index("end")
                text_widget.insert("end", seg_text)
                end_pos = text_widget.index("end")
                if seg_tag is not None:
                    text_widget.tag_add(seg_tag, start_pos, end_pos)

            text_widget.insert("end", "\n")

        # track line index (not strictly needed but could be extended if we wanted ref)
    # done

########################################
# MEMORY MANAGER
########################################

class MemoryManager:
    """
    Per-model chat memory.
    Stored as list[ { "role": "user"|"assistant", "content": "..."} ]

    We will *not* store system messages in memory file.
    We'll build the final Phi-4 prompt with a system turn at runtime.

    Microsoft positions Phi-4-mini-instruct as a chat-completion style model
    with explicit role tokens like <|system|>, <|user|>, and <|assistant|>,
    where each turn ends with <|end|>. This is how the model was tuned to
    follow instructions and safety alignment. :contentReference[oaicite:2]{index=2}
    """

    def __init__(self):
        self.cache = {}  # model_id -> message list

    def _path_for(self, model_id: str) -> str:
        safe_id = model_id.replace("/", "_").replace("\\", "_")
        return os.path.join(MEMORY_DIR, f"{safe_id}.json")

    def load(self, model_id: str):
        path = self._path_for(model_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    msgs = json.load(f)
                    if isinstance(msgs, list):
                        self.cache[model_id] = msgs
                        return
            except Exception:
                pass
        self.cache[model_id] = []

    def get_all(self, model_id: str):
        if model_id not in self.cache:
            self.load(model_id)
        return self.cache[model_id]

    def append(self, model_id: str, role: str, content: str):
        if model_id not in self.cache:
            self.load(model_id)
        self.cache[model_id].append({
            "role": role,
            "content": content
        })
        self._save(model_id)

    def _save(self, model_id: str):
        path = self._path_for(model_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cache[model_id], f, ensure_ascii=False, indent=2)

    def build_context_block(self, model_id: str, new_user_message: str) -> str:
        """
        Build Phi-4 style prompt.

        Official usage for Phi-4-mini-instruct is "chat completion":
        <|system|>...<|end|>
        <|user|>...<|end|>
        <|assistant|>...<|end|>
        ...
        <|user|>CURRENT QUESTION<|end|>
        <|assistant|>

        We STOP at <|assistant|> with nothing else. That makes the model
        continue naturally as the assistant, instead of talking to itself
        as both roles. This pattern (assistant prompt handoff) is widely
        recommended for Phi-4 chat-style inference. :contentReference[oaicite:3]{index=3}
        """

        system_msg = (
            "You are an FRC robotics programming assistant. "
            "You write WPILib / swerve / subsystem code with clean structure, "
            "odometry, Pigeon2/IMU usage, safe elevator logic, and Constants. "
            "You only respond as the assistant. "
            "Never pretend to be the user. "
            "Don't include <|user|> or <|assistant|> tags in your answer."
        )

        history = self.get_all(model_id)[-MAX_CONTEXT_MESSAGES:]

        parts = []
        parts.append(f"<|system|>{system_msg}<|end|>")

        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()

            if role == "user":
                parts.append(f"<|user|>{content}<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>{content}<|end|>")

        parts.append(f"<|user|>{new_user_message.strip()}<|end|>")
        parts.append("<|assistant|>")

        prompt = "\n".join(parts)
        return prompt

########################################
# MODEL RUNNER
########################################

class ModelRunner:
    """
    Handles loading base model + LoRA adapters for inference and generating output.
    We assume you already have a base model (Phi-4-mini-instruct)
    and maybe an adapter dir you trained (LoRA).
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # model_id options for dropdown
        # Base model alone
        # and optionally finetuned adapters you saved after training
        self.available_models = [
            {
                "label": "Phi-4-mini-instruct (base)",
                "base": "microsoft/Phi-4-mini-instruct",
                "adapter": None,
            },
            # add more entries here after finetune:
            # {
            #     "label": "My FRC Adapter Epoch1",
            #     "base": "microsoft/Phi-4-mini-instruct",
            #     "adapter": "./llm_tuned_any/epoch_1",
            # },
        ]

        # defaults
        self.current_index = 0
        self.current_model_id = self.available_models[self.current_index]["label"]

        self.tokenizer = None
        self.model = None

        # tunables
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P

        self._load_current_model()

    def list_model_labels(self):
        return [m["label"] for m in self.available_models]

    def select_model_by_label(self, label: str):
        # pick from available_models
        for i, m in enumerate(self.available_models):
            if m["label"] == label:
                self.current_index = i
                self.current_model_id = m["label"]
                self._load_current_model()
                return
        raise ValueError("Model label not found")

    def _load_current_model(self):
        cfg = self.available_models[self.current_index]
        base_name = cfg["base"]
        adapter_path = cfg["adapter"]

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            # Phi-4 tokenizer quirks: EOS / PAD tokens can map to special end tokens.
            # People have noted Phi-4 uses special <|end|> style tokens and care must be
            # taken so EOS and PAD aren't identical, to avoid infinite loops. :contentReference[oaicite:4]{index=4}
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load base model in half precision for inference
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if adapter_path:
            # attach LoRA adapter
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16,
            )
        else:
            self.model = base_model

        self.model.to(self.device)
        self.model.eval()

    def set_generation_params(self, max_new_tokens, temperature, top_p):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt: str) -> str:
        """
        Run a single completion.
        We'll:
        - tokenize prompt
        - generate
        - decode full
        - strip prompt prefix so we only keep new assistant text
        """

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(
            out_ids[0],
            skip_special_tokens=False  # so we can clean <|assistant|> / <|end|> manually
        )

        # slice off the prompt prefix if it's literally repeated
        if full_text.startswith(prompt):
            new_text = full_text[len(prompt):]
        else:
            new_text = full_text

        # cleanup leading "<|assistant|>" if model echoed it
        if new_text.strip().startswith("<|assistant|>"):
            new_text = new_text.strip()[len("<|assistant|>"):].lstrip()

        # cleanup trailing "<|end|>" etc.
        if new_text.rstrip().endswith("<|end|>"):
            new_text = new_text.rstrip()[:-len("<|end|>")].rstrip()

        # final sanitize
        return new_text.strip()

########################################
# MAIN APP UI
########################################

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FRC LLM Console")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI Semibold", 11))
        style.configure("TButton",
                        background="#2d2d30",
                        foreground="#ffffff",
                        borderwidth=0,
                        focusthickness=3,
                        focuscolor="none",
                        font=("Segoe UI", 10))
        style.map("TButton",
                  background=[("active", "#3a3a3d")])

        # left panel: model chooser + params + memory status
        self.sidebar = ttk.Frame(self.root, padding=12, style="TFrame")
        self.sidebar.pack(side="left", fill="y")

        ttk.Label(self.sidebar, text="MODEL", style="Header.TLabel").pack(anchor="w", pady=(0,4))
        self.runner = ModelRunner()

        self.model_var = tk.StringVar(value=self.runner.current_model_id)
        self.model_menu = ttk.Combobox(
            self.sidebar,
            textvariable=self.model_var,
            values=self.runner.list_model_labels(),
            state="readonly"
        )
        self.model_menu.pack(fill="x")
        self.model_menu.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Label(self.sidebar, text="Max new tokens", style="TLabel").pack(anchor="w", pady=(16,0))
        self.max_tokens_var = tk.IntVar(value=DEFAULT_MAX_NEW_TOKENS)
        self.max_tokens_scale = ttk.Scale(
            self.sidebar,
            from_=32,
            to=1024,
            orient="horizontal",
            command=self._on_tokens_scale
        )
        self.max_tokens_scale.set(DEFAULT_MAX_NEW_TOKENS)
        self.max_tokens_scale.pack(fill="x")

        ttk.Label(self.sidebar, text="Temperature", style="TLabel").pack(anchor="w", pady=(16,0))
        self.temp_var = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        self.temp_scale = ttk.Scale(
            self.sidebar,
            from_=0.1,
            to=1.5,
            orient="horizontal",
            command=self._on_temp_scale
        )
        self.temp_scale.set(DEFAULT_TEMPERATURE)
        self.temp_scale.pack(fill="x")

        ttk.Label(self.sidebar, text="Top-p", style="TLabel").pack(anchor="w", pady=(16,0))
        self.top_p_var = tk.DoubleVar(value=DEFAULT_TOP_P)
        self.top_p_scale = ttk.Scale(
            self.sidebar,
            from_=0.1,
            to=1.0,
            orient="horizontal",
            command=self._on_top_p_scale
        )
        self.top_p_scale.set(DEFAULT_TOP_P)
        self.top_p_scale.pack(fill="x")

        ttk.Label(self.sidebar, text="Memory", style="Header.TLabel").pack(anchor="w", pady=(24,4))
        self.mem = MemoryManager()
        self.mem.load(self.runner.current_model_id)

        self.memory_label = ttk.Label(
            self.sidebar,
            text=self._memory_summary(),
            style="TLabel",
            wraplength=200,
            justify="left"
        )
        self.memory_label.pack(fill="x")

        self.clear_mem_btn = ttk.Button(
            self.sidebar,
            text="Clear Memory",
            command=self._clear_memory_click
        )
        self.clear_mem_btn.pack(fill="x", pady=(8,0))

        # center panel: chat log
        center_frame = ttk.Frame(self.root, padding=12, style="TFrame")
        center_frame.pack(side="left", fill="both", expand=True)

        ttk.Label(center_frame, text="Chat", style="Header.TLabel").pack(anchor="w", pady=(0,6))

        self.chat_box = scrolledtext.ScrolledText(
            center_frame,
            wrap="word",
            bg="#252526",
            fg="#ffffff",
            insertbackground="#ffffff",
            font=("Segoe UI", 10),
            height=20
        )
        self.chat_box.pack(fill="both", expand=True)
        self.chat_box.configure(state="disabled")

        # bottom input area
        bottom_frame = ttk.Frame(center_frame, padding=(0,8,0,0), style="TFrame")
        bottom_frame.pack(fill="x")

        self.entry = tk.Text(
            bottom_frame,
            height=4,
            bg="#1e1e1e",
            fg="#ffffff",
            insertbackground="#ffffff",
            font=("Segoe UI", 10),
            wrap="word",
            relief="solid",
            bd=1
        )
        self.entry.pack(side="left", fill="both", expand=True, padx=(0,8))

        self.send_btn = ttk.Button(
            bottom_frame,
            text="Send",
            command=self._on_send_click
        )
        self.send_btn.pack(side="left")

        # "Generating..." animated label (shows while model is working)
        self.gen_status = ttk.Label(
            bottom_frame,
            text="",
            style="TLabel"
        )
        self.gen_status.pack(side="left", padx=(8,0))

        # state
        self.generating = False
        self._spinner_phase = 0

    ########################################
    # GUI HELPERS
    ########################################

    def _memory_summary(self):
        msgs = self.mem.get_all(self.runner.current_model_id)
        return f"{len(msgs)} turns stored"

    def _update_memory_label(self):
        self.memory_label.configure(text=self._memory_summary())

    def _append_chat(self, role: str, text: str):
        """
        Append a chat bubble to chat_box with lightweight styling + markdown.
        """
        self.chat_box.configure(state="normal")

        if role == "user":
            header = "You:\n"
            header_color = "#9cdcfe"  # light blue
        elif role == "assistant":
            header = "Assistant:\n"
            header_color = "#ce9178"  # light orange
        else:
            header = "System:\n"
            header_color = "#c586c0"  # purple

        # header
        start_header = self.chat_box.index("end")
        self.chat_box.insert("end", header)
        end_header = self.chat_box.index("end")
        self.chat_box.tag_add(f"hdr_{role}", start_header, end_header)
        self.chat_box.tag_config(f"hdr_{role}", foreground=header_color, font=("Segoe UI Semibold", 10))

        # body with markdown styling
        start_body_index = self.chat_box.index("end")
        apply_markdown_tags(self.chat_box, start_body_index, text.strip())
        self.chat_box.insert("end", "\n\n")

        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")

    def _spinner_tick(self):
        if not self.generating:
            self.gen_status.configure(text="")
            return
        dots = [".  ", ".. ", "..."]
        self.gen_status.configure(text="Generating" + dots[self._spinner_phase % len(dots)])
        self._spinner_phase += 1
        self.root.after(400, self._spinner_tick)

    ########################################
    # EVENTS
    ########################################

    def _on_model_change(self, event=None):
        new_label = self.model_var.get()
        try:
            self.runner.select_model_by_label(new_label)
            # reload memory for that model
            self.mem.load(self.runner.current_model_id)
            self._update_memory_label()
            self._append_chat("system", f"Loaded model: {new_label}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed loading model:\n{e}")

    def _on_tokens_scale(self, val):
        try:
            v = int(float(val))
            self.max_tokens_var.set(v)
            self.runner.set_generation_params(
                max_new_tokens=v,
                temperature=self.temp_var.get(),
                top_p=self.top_p_var.get()
            )
        except:
            pass

    def _on_temp_scale(self, val):
        try:
            v = float(val)
            self.temp_var.set(v)
            self.runner.set_generation_params(
                max_new_tokens=self.max_tokens_var.get(),
                temperature=v,
                top_p=self.top_p_var.get()
            )
        except:
            pass

    def _on_top_p_scale(self, val):
        try:
            v = float(val)
            self.top_p_var.set(v)
            self.runner.set_generation_params(
                max_new_tokens=self.max_tokens_var.get(),
                temperature=self.temp_var.get(),
                top_p=v
            )
        except:
            pass

    def _clear_memory_click(self):
        confirm = messagebox.askyesno(
            "Clear Memory",
            "Delete all remembered conversation for this model?"
        )
        if not confirm:
            return
        self.mem.cache[self.runner.current_model_id] = []
        self.mem._save(self.runner.current_model_id)
        self._update_memory_label()
        self._append_chat("system", "Memory cleared.")

    def _on_send_click(self):
        if self.generating:
            return
        user_text = self.entry.get("1.0", "end").strip()
        if not user_text:
            return

        # show user msg in chat right now
        self._append_chat("user", user_text)

        # save user msg into memory
        self.mem.append(self.runner.current_model_id, "user", user_text)
        self._update_memory_label()

        self.entry.delete("1.0", "end")

        # now run generation in thread
        self.generating = True
        self._spinner_phase = 0
        self._spinner_tick()

        t = threading.Thread(target=self._generate_and_display, args=(user_text,))
        t.daemon = True
        t.start()

    ########################################
    # GENERATION FLOW
    ########################################

    def _generate_and_display(self, user_text: str):
        try:
            # build Phi-4-style prompt with context + system
            prompt = self.mem.build_context_block(
                self.runner.current_model_id,
                user_text
            )

            # run model
            reply = self.runner.generate(prompt)

            # clean reply (strip leftover assistant tags or <|end|>)
            reply_clean = reply.strip()
            if reply_clean.startswith("<|assistant|>"):
                reply_clean = reply_clean[len("<|assistant|>"):].lstrip()
            if reply_clean.endswith("<|end|>"):
                reply_clean = reply_clean[:-len("<|end|>")].rstrip()

            # save assistant msg to memory
            self.mem.append(self.runner.current_model_id, "assistant", reply_clean)
            self._update_memory_label()

            def finish():
                self._append_chat("assistant", reply_clean)

            self.root.after(0, finish)

        except Exception as e:
            def fail():
                self._append_chat("system", f"❌ Generation error: {e}")
            self.root.after(0, fail)

        finally:
            self.generating = False

########################################
# MAIN LAUNCH
########################################

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
