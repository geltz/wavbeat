#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog
import tkinterdnd2 as tkdnd
import threading
import os

# Import wavbeat processing
try:
    from wavbeat import main as wavbeat_process
except ImportError:
    # This is a fallback for when the wavbeat module is not installed
    # It allows the GUI to run for demonstration purposes.
    def wavbeat_process(**kwargs):
        print("Received parameters:")
        for key, value in kwargs.items():
            print(f"- {key}: {value:.2f}" if isinstance(value, float) else f"- {key}: {value}")
        print("Simulating audio processing... done.")
        import time
        time.sleep(2) # Simulate work

class GradientFrame(tk.Canvas):
    """A canvas that draws a vertical gradient background."""
    def __init__(self, parent, color1, color2, **kwargs):
        super().__init__(parent, **kwargs)
        self.color1 = color1
        self.color2 = color2
        self.bind("<Configure>", self._draw_gradient)

    def _draw_gradient(self, event=None):
        self.delete("gradient")
        width = self.winfo_width()
        height = self.winfo_height()
        if width == 1 or height == 1: # Avoid drawing on tiny canvas
            return
            
        limit = height
        (r1, g1, b1) = self.winfo_rgb(self.color1)
        (r2, g2, b2) = self.winfo_rgb(self.color2)
        r_ratio = float(r2 - r1) / limit
        g_ratio = float(g2 - g1) / limit
        b_ratio = float(b2 - b1) / limit

        for i in range(limit):
            nr = int(r1 + (r_ratio * i))
            ng = int(g1 + (g_ratio * i))
            nb = int(b1 + (b_ratio * i))
            color = f'#{nr>>8:02x}{ng>>8:02x}{nb>>8:02x}'
            self.create_line(0, i, width, i, tags=("gradient",), fill=color)

class GradientText(tk.Canvas):
    """A canvas that draws text with a smooth horizontal gradient."""
    def __init__(self, parent, text, font, color1, color2, **kwargs):
        super().__init__(parent, **kwargs)
        self.text = text
        self.font = font
        self.color1 = color1
        self.color2 = color2
        self.bind("<Configure>", self._draw_text)

    def _draw_text(self, event=None):
        self.delete("gradient_text")
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:
            return

        (r1, g1, b1) = self.winfo_rgb(self.color1)
        (r2, g2, b2) = self.winfo_rgb(self.color2)
        
        # Measure the text width to center it properly
        temp_id = self.create_text(0, 0, text=self.text, font=self.font, anchor='w')
        bbox = self.bbox(temp_id)
        self.delete(temp_id)
        text_width = bbox[2] - bbox[0]
        x_start = (width - text_width) / 2
        
        # Draw each character with an interpolated color for a horizontal gradient
        for i, char in enumerate(self.text):
            # Calculate the color for the current character
            ratio = i / (len(self.text) - 1) if len(self.text) > 1 else 0
            nr = int(r1 + (r2 - r1) * ratio)
            ng = int(g1 + (g2 - g1) * ratio)
            nb = int(b1 + (b2 - b1) * ratio)
            color = f'#{nr>>8:02x}{ng>>8:02x}{nb>>8:02x}'
            
            # Draw the character and get its width to position the next one
            char_id = self.create_text(
                x_start, height / 2, text=char, font=self.font,
                fill=color, anchor='w', tags=("gradient_text",)
            )
            char_bbox = self.bbox(char_id)
            char_width = char_bbox[2] - char_bbox[0] if char_bbox else 0
            x_start += char_width

class WavbeatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("wavbeat")
        self.root.geometry("540x600")
        self.root.resizable(False, False)
        
        # Soft pastel lavender color theme
        self.bg = "#E6E6FA"  # Lavender
        self.fg = "#483D8B"  # DarkSlateBlue
        self.accent = "#9370DB"  # MediumPurple
        self.drop_bg = "#FFFFFF" # White
        self.grad_start = "#C1A2E6"
        self.grad_end = self.accent
        
        # Font
        self.font_family = "Segoe UI"

        self.root.configure(bg=self.bg)
        self.processing = False
        
        # To store original parameter ranges for scaling
        self.param_ranges = {}
        self.param_steps = {}
        self.param_is_int = {}
        self.param_label_vars = {}

        self._build_ui()
    
    def _build_ui(self):
            # Main container with gradient background
            main = GradientFrame(self.root, color1=self.bg, color2="#F0F0FF", highlightthickness=0, bg=self.bg)
            main.pack(fill=tk.BOTH, expand=True)
            
            # Title with gradient text
            title_canvas = GradientText(main, "wavbeat", (self.font_family, 28, "bold"),
                                        self.grad_start, self.grad_end, bg=self.bg, 
                                        highlightthickness=0)
            main.create_window(270, 30, window=title_canvas, width=250, height=44)

            # Drop zone
            self.drop_frame = tk.Frame(main, bg=self.drop_bg,
                                       highlightbackground=self.accent,
                                       highlightthickness=1)
            main.create_window(270, 135, window=self.drop_frame, width=460, height=150)

            # --- AESTHETIC CHANGE: Made drop-zone text larger and added icon ---
            drop_label = tk.Label(self.drop_frame,
                                  text="♫\ndrag audio file here\n(or click to browse)",
                                  font=(self.font_family, 14), bg=self.drop_bg, fg="#6A5ACD",
                                  cursor="hand2")
            drop_label.pack(expand=True, fill=tk.BOTH)
            drop_label.bind("<Button-1>", lambda e: self._browse_file())

            # Register drop target
            self.drop_frame.drop_target_register(tkdnd.DND_FILES)
            self.drop_frame.dnd_bind('<<Drop>>', self._on_drop)

            
            # Status Label
            self.status_label = tk.Label(main, text="ready", 
                                         font=(self.font_family, 9), 
                                         bg="#F0F0FF", fg="#666")
            main.create_window(270, 220, window=self.status_label)
            
            # Parameters frame
            param_frame = tk.Frame(main, bg=self.bg)
            main.create_window(270, 420, window=param_frame, width=460)

            # --- AESTHETIC CHANGE: Use .grid() layout for clean columns ---
            # Configure grid columns for alignment
            # Column 0: Label
            # Column 1: Slider (expands)
            # Column 2: Value
            param_frame.columnconfigure(0, weight=2, uniform="params")
            param_frame.columnconfigure(1, weight=3, uniform="params")
            param_frame.columnconfigure(2, weight=1, uniform="params")

            # Parameters
            self.params = {}

            configs = [
                ("bpm", "bpm", 120, 60, 200, 1),
                ("speed", "speed", 1.0, 0.5, 2.0, 0.1),
                ("rate (chops)", "rate", 1.0, 0.1, 2.0, 0.1),
                # --- REVERB CHANGE: Default changed from 1.0 to 1.2 ---
                ("reverb", "reverb", 1.2, 0.0, 2.0, 0.1),
                ("bars", "bars", 8, 1, 32, 1),
                ("subdiv", "subdiv", 1, 1, 8, 1),
                ("hat density", "hat_density", 0.60, 0.0, 1.0, 0.05),
                ("clap dev prob", "clap_dev_prob", 0.10, 0.0, 1.0, 0.05),
                ("clap dev (ms)", "clap_dev_ms", 22.0, 0.0, 100.0, 1.0),
            ]

            for i, (label, key, default, min_val, max_val, step) in enumerate(configs):
                # --- LAYOUT CHANGE: Removed intermediary 'row' frame ---

                # Store metadata
                self.param_ranges[key] = {'min': min_val, 'max': max_val}
                self.param_steps[key] = step
                self.param_is_int[key] = isinstance(default, int)

                # Use DoubleVar and real ranges
                var = tk.DoubleVar(value=float(default))
                self.params[key] = var

                # --- LAYOUT CHANGE: Place label in grid column 0 ---
                lbl = tk.Label(param_frame, text=label, font=(self.font_family, 10),
                               bg=self.bg, fg=self.fg, anchor="w")
                lbl.grid(row=i, column=0, sticky="w", padx=(0, 10), pady=5)

                # --- LAYOUT CHANGE: Place scale in grid column 1 ---
                scale = ttk.Scale(param_frame, from_=min_val, to=max_val,
                                  variable=var, orient=tk.HORIZONTAL,
                                  command=lambda _val, k=key: self._on_slider(k))
                scale.grid(row=i, column=1, sticky="ew", padx=5, pady=5) # 'ew' = expand horizontally

                # --- LAYOUT CHANGE: Place value label in grid column 2 ---
                sval = tk.StringVar()
                self.param_label_vars[key] = sval
                val_label = tk.Label(param_frame, textvariable=sval, font=(self.font_family, 9),
                                     bg=self.bg, fg="#6A5ACD", width=8, anchor="e")
                val_label.grid(row=i, column=2, sticky="e", padx=(10, 0), pady=5) # 'e' = align right

            # --- LAYOUT CHANGE: Place clap toggle on the grid ---
            next_row = len(configs)
            self.clap_var = tk.BooleanVar(value=False)
            clap_cb = tk.Checkbutton(param_frame, text="enable claps",
                                     variable=self.clap_var,
                                     font=(self.font_family, 10),
                                     bg=self.bg, fg=self.fg,
                                     selectcolor=self.drop_bg,
                                     activebackground=self.bg,
                                     activeforeground=self.fg,
                                     bd=0,
                                     highlightthickness=0,
                                     anchor="w")
            clap_cb.grid(row=next_row, column=0, columnspan=2, sticky="w", pady=5)

            # Style for sliders (unchanged)
            style = ttk.Style()
            style.theme_use('clam')
            style.configure("Horizontal.TScale",
                            background=self.bg,
                            troughcolor="#F0F0FF",
                            borderwidth=0,
                            lightcolor=self.accent,
                            darkcolor=self.accent,
                            sliderlength=20)
            style.map("Horizontal.TScale", background=[('active', self.bg)])

            # Initialize labels once everything exists
            for key in self.params.keys():
                self._on_slider(key)


    def _format_val(self, key, val):
        step = self.param_steps[key]
        is_int = self.param_is_int[key]
        if is_int or step >= 1:
            return str(int(round(val)))
        # derive sensible decimals from step (e.g., 0.05 -> 2 dp)
        s = f"{step:.10f}".rstrip('0').rstrip('.')
        decimals = len(s.split('.')[1]) if '.' in s else 0
        return f"{val:.{decimals}f}"

    def _on_slider(self, key):
        """Snap the slider to the configured step and clamp, then update its label."""
        var = self.params[key]
        v = float(var.get())
        min_v = self.param_ranges[key]['min']
        max_v = self.param_ranges[key]['max']
        step = self.param_steps[key]

        # snap relative to min (avoids drift)
        snapped = round((v - min_v) / step) * step + min_v
        # clamp
        snapped = max(min(snapped, max_v), min_v)

        # avoid feedback loops unless needed
        if abs(snapped - v) > 1e-9:
            var.set(snapped)

        # label
        self.param_label_vars[key].set(self._format_val(key, float(var.get())))


    def _browse_file(self):
        file = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.aiff"), 
                      ("All files", "*.*")]
        )
        if file:
            self._process_file(file)
    
    def _on_drop(self, event):
        # Clean up file path from tkdnd
        file = event.data.strip('{}')
        if file:
            self._process_file(file)
    
    def _process_file(self, filepath):
        if self.processing:
            self.status_label.config(text="already processing...", fg="#E03B8B")
            return
        
        if not os.path.exists(filepath):
            self.status_label.config(text="file not found", fg="#E03B8B")
            return
            
        self.processing = True
        self.status_label.config(text=f"processing: {os.path.basename(filepath)}", 
                                fg=self.accent)
        
        thread = threading.Thread(target=self._run_wavbeat, args=(filepath,))
        thread.daemon = True
        thread.start()
    
    def _run_wavbeat(self, filepath):
        try:
            args = {"input_file": filepath}
            for key, var in self.params.items():
                val = float(var.get())
                if self.param_is_int[key]:
                    args[key] = int(round(val))
                else:
                    args[key] = val

            args["clap"] = self.clap_var.get()

            wavbeat_process(**args)
            self.root.after(0, self._set_status, "✓ complete", "#20B2AA")
        except Exception as e:
            self.root.after(0, self._set_status, f"error: {str(e)[:30]}", "#E03B8B")
        finally:
            self.processing = False
    
    def _set_status(self, text, color):
        self.status_label.config(text=text, fg=color)
        self.root.after(4000, lambda: self.status_label.config(text="ready", fg="#666"))

if __name__ == "__main__":
    # Use tkinterdnd2's Tk object as the root
    root = tkdnd.Tk()
    app = WavbeatGUI(root)
    root.mainloop()