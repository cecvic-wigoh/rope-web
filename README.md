<img width="2343" height="1375" alt="Screenshot 2026-07-11 132434" src="https://github.com/user-attachments/assets/abb3cfe1-9ec6-4ce5-b8fe-8c9de7a45294" />

Rope implements the insightface inswapper_128 model with a helpful GUI.
### [Discord](https://discord.gg/EcdVAFJzqp)

### [Donate](https://www.paypal.com/donate/?hosted_button_id=Y5SB9LSXFGRF2)

### ${{\color{Goldenrod}{\textsf{Last Updated 2026-07-11}}}}$ ###
### ${{\color{Goldenrod}{\textsf{Welcome to Rope-Bronze!}}}}$ ###

### Updates for Rope-Bronze: ###
* New, more responsive UI
* TRT Engine for better performance
* Batched inswapper for better 256 and 512 mode performance
* Settings tab for managing folders, models threading, benchmarking, ...
* New Likeness / Fidelity settings
* Color Matching (LAB) for accureate color matching
* XSeg masker
* Easier Embedding management. Drag and drop embeddings to reorder them.
* New Capture mode. Move and resize a window on your desktop to swap whatever is in it.

### Install from scratch:
```cmd
py -3.12 -m venv venv
```
```cmd
venv\Scripts\activate
```
```cmd
pip install -r requirements.lock.txt
```
Also, copy models from the Rope-Bronze Models Release to somewhere on your drive. In settings, select the folder they were copied to (you have to unzip them).

On macOS / Linux use `python3.12 -m venv venv`, `source venv/bin/activate`, and launch with `./rope.sh`. Note: the swap pipeline currently requires an NVIDIA GPU (CUDA); on other machines the UI runs but swapping does not (see [ANALYSIS.md](ANALYSIS.md)).

### Web app (Lightning AI / headless GPU servers): ###
Run Rope in the browser on a remote NVIDIA GPU. See [webapp/README.md](webapp/README.md):
```bash
bash webapp/setup_lightning.sh && python webapp/app.py
```

### Command line: ###
```
python Rope.py [--config-dir DIR] [--models-dir DIR] [--output-dir DIR]
               [--preset NAME_OR_FILE] [--stylesheet QSS] [--no-backend] [--print-config]
```
`Rope.bat` and `rope.sh` forward these flags. Environment variables: `ROPE_HOME` (config dir), `ROPE_MODELS` (default models folder).

### Customization: ###
All user files live in the config dir (`--config-dir`, else `$ROPE_HOME`, else the repo root). Run `python Rope.py --print-config` to see where.

| File | Purpose |
|---|---|
| `data.json` | Folders, window layout, shortcuts, fonts |
| `saved_parameters.json` | Quick-save slot (Save Params / Load Params / Ctrl+S) |
| `presets/<name>.json` | Named presets, managed from the **Preset** row in the Parameters tab (Ctrl+Shift+S = Save As) |
| `user.qss` | Optional Qt stylesheet appended after the built-in theme |

**Keyboard shortcuts.** Override any of these in `data.json` (edit while Rope is closed). An empty string disables a shortcut:
```json
"shortcuts": { "play_pause": "P", "nudge_back": "J", "nudge_forward": "L", "toggle_hud": "" },
"nudge_frames": 10
```
Actions and their defaults: `play_pause` Space, `timeline_start` Q, `nudge_back` A, `nudge_forward` D, `frame_back` Left, `frame_forward` Right, `seek_start` Home, `seek_end` End, `add_marker` M, `delete_marker` Shift+M, `prev_marker` Shift+, , `next_marker` Shift+. , `save_params` Ctrl+S, `save_preset_as` Ctrl+Shift+S, `toggle_hud` F3.

**Fonts.** `"ui_font_family": "Inter", "ui_font_size": 10` in `data.json`.

**Theme.** For example, a `user.qss` with a different accent color:
```css
QPushButton:checked, QPushButton[state="on"] { color: #4FC3F7; }
QTabBar::tab:selected { color: #4FC3F7; }
```

Hovering any button or slider shows its help text in the bottom status bar. Status and error messages appear there too.

### Disclaimer: ###
Rope is a personal project that I'm making available to the community as a thank you for all of the contributors ahead of me.
I've copied the disclaimer from [Swap-Mukham](https://github.com/harisreedhar/Swap-Mukham) here since it is well-written and applies 100% to this repo.
 
I would like to emphasize that our swapping software is intended for responsible and ethical use only. I must stress that users are solely responsible for their actions when using our software.

Intended Usage: This software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. I encourage users to explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.

Ethical Guidelines: Users are expected to adhere to a set of ethical guidelines when using our software. These guidelines include, but are not limited to:

Not creating or sharing content that could harm, defame, or harass individuals. Obtaining proper consent and permissions from individuals featured in the content before using their likeness. Avoiding the use of this technology for deceptive purposes, including misinformation or malicious intent. Respecting and abiding by applicable laws, regulations, and copyright restrictions.

Privacy and Consent: Users are responsible for ensuring that they have the necessary permissions and consents from individuals whose likeness they intend to use in their creations. We strongly discourage the creation of content without explicit consent, particularly if it involves non-consensual or private content. It is essential to respect the privacy and dignity of all individuals involved.

Legal Considerations: Users must understand and comply with all relevant local, regional, and international laws pertaining to this technology. This includes laws related to privacy, defamation, intellectual property rights, and other relevant legislation. Users should consult legal professionals if they have any doubts regarding the legal implications of their creations.

Liability and Responsibility: We, as the creators and providers of the deep fake software, cannot be held responsible for the actions or consequences resulting from the usage of our software. Users assume full liability and responsibility for any misuse, unintended effects, or abusive behavior associated with the content they create.

By using this software, users acknowledge that they have read, understood, and agreed to abide by the above guidelines and disclaimers. We strongly encourage users to approach this technology with caution, integrity, and respect for the well-being and rights of others.

Remember, technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of deep fake technology for the betterment of society.



  
