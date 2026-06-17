"""
TTI_main.py — unified entry point for the TTI system.
"""
from __future__ import annotations
import argparse, sys, time, copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from TTI_config import TTIConfig, get_config, update_config, reset_config
from TTI_core   import TTIImage, ImageCanvas, ImageIO, ColorUtils, ImageValidator, TTIError, TTIIOError
from TTI_art    import ProceduralArt, VisualEffects, StreamingWriter, AnimationEngine, CustomBitDepth
from TTI_ai     import TTIGenerator, NLPAnalyser, PromptAnalysis


class TTIPipeline:
    """Top-level façade wiring config, AI, art and I/O together."""

    _ART_TYPES = {
        "mandelbrot","julia","sierpinski","plasma","voronoi","noise",
        "gradient","radial","checkerboard","waves","circles","spiral",
    }
    _EFFECTS = {
        "blur","gaussian_blur","sharpen","edge","emboss","grayscale",
        "sepia","invert","noise","pixelate","vignette","brightness","contrast",
    }

    def __init__(self, config: Optional[TTIConfig] = None) -> None:
        self.cfg  = config or get_config()
        self.cfg.ensure_dirs()
        self._gen = TTIGenerator(self.cfg)
        self._nlp = NLPAnalyser(self.cfg)

    # ── Generation ────────────────────────────────────────────────────

    def generate(self, prompt, output=None, width=None, height=None,
                 seed=None, fmt=None):
        img = self._gen.generate(prompt, width=width, height=height, seed=seed)
        if output:
            saved = img.save(output, fmt=fmt)
            self._log(f"Saved → {saved}")
        return img

    def generate_batch(self, prompts, output_dir="tti_output", width=None,
                       height=None, seed=None, fmt="png", prefix="img"):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, prompt in enumerate(prompts):
            self._log(f"[{i+1}/{len(prompts)}] {prompt[:50]}")
            img  = self._gen.generate(prompt, width=width, height=height,
                                      seed=seed+i if seed else None)
            path = out / f"{prefix}_{i:04d}.{fmt}"
            img.save(path); paths.append(path)
        return paths

    def generate_variations(self, prompt, n=4, output_dir="tti_output",
                            width=None, height=None, fmt="png"):
        imgs  = self._gen.generate_variations(prompt, n_variations=n, width=width, height=height)
        out   = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(imgs):
            path = out / f"variation_{i:04d}.{fmt}"
            img.save(path); paths.append(path)
        self._log(f"Saved {len(paths)} variations → {out}")
        return paths

    def interpolate(self, prompt_a, prompt_b, steps=6, output_dir="tti_output",
                    width=None, height=None, fmt="png"):
        imgs  = self._gen.interpolate(prompt_a, prompt_b, steps, width, height)
        out   = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(imgs):
            path = out / f"interp_{i:04d}.{fmt}"
            img.save(path); paths.append(path)
        self._log(f"Interpolation ({steps} frames) → {out}")
        return paths

    def analyse(self, prompt):
        a = self._nlp.analyse(prompt)
        self._log(str(a))
        return a

    # ── Procedural art ────────────────────────────────────────────────

    def art(self, art_type, output=None, width=None, height=None,
            seed=None, **kwargs):
        w, h, cfg = (width or self.cfg.image.default_width,
                     height or self.cfg.image.default_height, self.cfg)
        dispatch = {
            "mandelbrot":   lambda: ProceduralArt.mandelbrot_set(w, h, config=cfg),
            "julia":        lambda: ProceduralArt.julia_set(
                                w, h,
                                c_real=kwargs.get("c_real", -0.7),
                                c_imag=kwargs.get("c_imag", 0.27015),
                                config=cfg),
            "sierpinski":   lambda: ProceduralArt.sierpinski_triangle(
                                w, h, depth=kwargs.get("depth", 7), config=cfg),
            "plasma":       lambda: ProceduralArt.plasma(w, h, config=cfg),
            "voronoi":      lambda: ProceduralArt.voronoi(
                                w, h,
                                n_cells=kwargs.get("n_cells", 20),
                                seed=seed, config=cfg),
            "noise":        lambda: ProceduralArt.perlin_noise_image(
                                w, h,
                                octaves=kwargs.get("octaves", 4), config=cfg),
            "gradient":     lambda: VisualEffects.create_linear_gradient(
                                w, h,
                                kwargs.get("c1", (70,130,200)),
                                kwargs.get("c2", (200,80,120)), config=cfg),
            "radial":       lambda: VisualEffects.create_radial_gradient(
                                w, h,
                                kwargs.get("center_color", (255,220,50)),
                                kwargs.get("edge_color",   (30,30,120)), config=cfg),
            "checkerboard": lambda: self._checkerboard(
                                w, h,
                                kwargs.get("square_size", 40),
                                kwargs.get("c1", (255,255,255)),
                                kwargs.get("c2", (30,30,30))),
            "waves":        lambda: self._waves(w, h, kwargs.get("freq", 0.05)),
            "circles":      lambda: self._concentric_circles(w, h, kwargs.get("n", 12)),
            "spiral":       lambda: self._spiral(w, h, kwargs.get("turns", 6)),
        }
        if art_type not in dispatch:
            raise TTIError(f"Unknown art type '{art_type}'. Valid: {sorted(self._ART_TYPES)}")
        img = dispatch[art_type]()
        if output:
            saved = img.save(output); self._log(f"Art saved → {saved}")
        return img

    def _checkerboard(self, w, h, sq, c1, c2):
        img = TTIImage(w, h, 24, config=self.cfg)
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        mask = ((xs//sq)+(ys//sq))%2==0
        arr  = np.where(mask[:,:,None], np.array(c1,dtype=np.uint8), np.array(c2,dtype=np.uint8))
        img.from_array(arr.astype(np.uint8)); return img

    def _waves(self, w, h, freq=0.05):
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        v = (np.sin(xs*freq)+np.sin(ys*freq*0.7))*0.5+0.5
        r = np.clip(v*100+50,  0,255).astype(np.uint8)
        g = np.clip(v*150+80,  0,255).astype(np.uint8)
        b = np.clip(v*200+100, 0,255).astype(np.uint8)
        img = TTIImage(w, h, 24, config=self.cfg)
        img.from_array(np.stack([r,g,b],axis=-1)); return img

    def _concentric_circles(self, w, h, n=12):
        img = TTIImage(w, h, 24, background=(10,10,40), config=self.cfg)
        canvas = ImageCanvas(img)
        cx,cy,max_r = w//2, h//2, min(w,h)//2-5
        for i in range(n):
            t = i/n; r = max(2,int(max_r*(1-t)))
            canvas.circle(cx, cy, r, ColorUtils.hsv_to_rgb(t,0.8,0.9), filled=False)
        return img

    def _spiral(self, w, h, turns=6):
        img = TTIImage(w, h, 24, background=(5,5,20), config=self.cfg)
        cx,cy,steps = w//2, h//2, turns*360
        for i in range(steps):
            t=i/steps; angle=t*turns*2*np.pi; radius=t*min(w,h)//2
            x=cx+int(radius*np.cos(angle)); y=cy+int(radius*np.sin(angle))
            col=ColorUtils.hsv_to_rgb(t,0.9,1.0)
            img.set_pixel(x,y,col); img.set_pixel(x+1,y,col); img.set_pixel(x,y+1,col)
        return img

    # ── Effects ───────────────────────────────────────────────────────

    def effect(self, effect_name, input_path, output=None, **kwargs):
        img = ImageIO.load(input_path, config=self.cfg)
        out = self._apply_effect(img, effect_name, **kwargs)
        if output:
            saved = out.save(output); self._log(f"Effect '{effect_name}' → {saved}")
        return out

    def _apply_effect(self, img, name, **kw):
        dispatch = {
            "blur":       lambda: VisualEffects.apply_blur(img, kw.get("radius")),
            "gaussian_blur": lambda: VisualEffects.apply_gaussian_blur(img, kw.get("sigma",2.0)),
            "sharpen":    lambda: VisualEffects.apply_sharpen(img),
            "edge":       lambda: VisualEffects.apply_edge_detect(img),
            "emboss":     lambda: VisualEffects.apply_emboss(img),
            "grayscale":  lambda: VisualEffects.apply_grayscale(img),
            "sepia":      lambda: VisualEffects.apply_sepia(img),
            "invert":     lambda: VisualEffects.apply_invert(img),
            "noise":      lambda: VisualEffects.add_noise(img, kw.get("intensity")),
            "pixelate":   lambda: VisualEffects.pixelate(img, kw.get("block_size",10)),
            "vignette":   lambda: VisualEffects.vignette(img, kw.get("strength",0.6)),
            "brightness": lambda: VisualEffects.adjust_brightness(img, kw.get("factor",1.2)),
            "contrast":   lambda: VisualEffects.adjust_contrast(img, kw.get("factor",1.2)),
        }
        if name not in dispatch:
            raise TTIError(f"Unknown effect '{name}'. Valid: {sorted(self._EFFECTS)}")
        return dispatch[name]()

    # ── Animation ─────────────────────────────────────────────────────

    def animate(self, prompt, n_frames=24, output_dir="tti_output/animation",
                width=None, height=None, fmt="png"):
        w,h  = (width or self.cfg.image.default_width,
                 height or self.cfg.image.default_height)
        out  = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        paths=[]; base=int(time.time())&0xFFFF
        for i in range(n_frames):
            img  = self._gen.generate(prompt, w, h, seed=base+i)
            path = out/f"frame_{i:04d}.{fmt}"
            img.save(path); paths.append(path)
            if self.cfg.log.show_progress: print(f"  Frame {i+1}/{n_frames}",end="\r")
        print(); self._log(f"Animation ({n_frames} frames) → {out}")
        return paths

    # ── Streaming ─────────────────────────────────────────────────────

    def stream_large(self, output, width, height, pattern="gradient"):
        path=Path(output); rng=np.random.default_rng(42)
        def row_gradient(y):
            g=int(y/height*255)
            return [(int(x/width*255),g,128) for x in range(width)]
        def row_checker(y):
            sq=max(1,min(width,height)//20)
            return [(255,255,255) if ((x//sq)+(y//sq))%2==0 else (30,30,30)
                    for x in range(width)]
        def row_noise(y):
            return [(int(rng.integers(0,256)),int(rng.integers(0,256)),
                     int(rng.integers(0,256))) for _ in range(width)]
        fn={"gradient":row_gradient,"checkerboard":row_checker,"noise":row_noise}.get(pattern,row_gradient)
        self._log(f"Streaming {width}×{height} BMP → {path}")
        with StreamingWriter(path, width, height, config=self.cfg) as w_:
            w_.generate_rows(fn)
        self._log(f"Done → {path}"); return path

    # ── Custom bit-depth ──────────────────────────────────────────────

    def custom_bitdepth(self, width, height, bits_per_channel=16,
                        num_channels=3, output=None, preview=None):
        img=CustomBitDepth(width,height,bits_per_channel,num_channels)
        mx=img.max_value
        for y in range(height):
            for x in range(width):
                vals=[int(x/width*mx),int(y/height*mx)]+[mx//(2+i) for i in range(num_channels-2)]
                img.set_pixel(x,y,vals)
        if output: img.save(output); self._log(f"CustomBitDepth → {output}")
        if preview: img.to_tti_image(self.cfg).save(preview); self._log(f"Preview → {preview}")
        return img

    # ── Config ────────────────────────────────────────────────────────

    def set_config(self, **kwargs): update_config(**kwargs)
    def show_config(self): print(self.cfg.to_json())
    def save_config(self, path="tti_config.json"):
        self.cfg.save(path); self._log(f"Config saved → {path}")
    def model_info(self): return self._gen.get_model_info()

    def _log(self, msg):
        if self.cfg.log.show_progress: print(f"[TTI] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# demo()
# ─────────────────────────────────────────────────────────────────────────────

def demo(output_dir="tti_demo"):
    import warnings; warnings.filterwarnings("ignore")
    print("="*65)
    print("  TTI — Text-To-Image System  |  Full Demo")
    print("="*65)
    cfg=get_config()
    cfg.image.default_width=400; cfg.image.default_height=400
    cfg.log.show_progress=True
    pipe=TTIPipeline(cfg)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    results=[]

    # AI generation
    ai_prompts=[
        ("blue ocean at sunset",            "ocean_sunset.png"),
        ("dark mysterious forest at night", "dark_forest.png"),
        ("bright golden mandelbrot fractal","mandelbrot_ai.png"),
        ("neon city lights reflection",     "neon_city.png"),
        ("red fire flame glowing",          "fire.png"),
        ("peaceful green valley",           "valley.png"),
        ("cold icy blue space nebula",      "nebula.png"),
        ("warm vintage desert landscape",   "desert.png"),
    ]
    print(f"\n── AI Text-to-Image ({len(ai_prompts)} prompts) ──")
    for prompt,fname in ai_prompts:
        try:
            img=pipe.generate(prompt,seed=42)
            img.save(out/fname); results.append((f"AI:{prompt}",out/fname))
        except Exception as e: print(f"  [!] {prompt}: {e}")

    # Procedural art
    art_specs=[
        ("mandelbrot",   {}, "art_mandelbrot.png"),
        ("julia",        {"c_real":-0.4,"c_imag":0.6}, "art_julia.png"),
        ("sierpinski",   {"depth":6}, "art_sierpinski.png"),
        ("plasma",       {}, "art_plasma.png"),
        ("voronoi",      {"n_cells":25}, "art_voronoi.png"),
        ("noise",        {"octaves":5}, "art_noise.png"),
        ("gradient",     {"c1":(255,80,80),"c2":(80,80,255)}, "art_gradient.png"),
        ("radial",       {"center_color":(255,220,0),"edge_color":(20,0,80)}, "art_radial.png"),
        ("checkerboard", {"square_size":30}, "art_checker.png"),
        ("waves",        {"freq":0.04}, "art_waves.png"),
        ("circles",      {"n":14}, "art_circles.png"),
        ("spiral",       {"turns":8}, "art_spiral.png"),
    ]
    print(f"\n── Procedural Art ({len(art_specs)} types) ──")
    for art_type,kwargs,fname in art_specs:
        try:
            img=pipe.art(art_type,**kwargs)
            img.save(out/fname); results.append((f"Art:{art_type}",out/fname))
        except Exception as e: print(f"  [!] {art_type}: {e}")

    # Effects
    print("\n── Visual Effects ──")
    base_img=pipe.art("plasma")
    fx_specs=[
        ("blur",{"radius":4},"fx_blur.png"),("sharpen",{},"fx_sharpen.png"),
        ("edge",{},"fx_edge.png"),("emboss",{},"fx_emboss.png"),
        ("grayscale",{},"fx_grayscale.png"),("sepia",{},"fx_sepia.png"),
        ("invert",{},"fx_invert.png"),("noise",{"intensity":0.2},"fx_noise.png"),
        ("pixelate",{"block_size":12},"fx_pixelate.png"),
        ("vignette",{"strength":0.7},"fx_vignette.png"),
        ("brightness",{"factor":1.5},"fx_bright.png"),
        ("contrast",{"factor":1.8},"fx_contrast.png"),
    ]
    for eff,kwargs,fname in fx_specs:
        try:
            img=pipe._apply_effect(base_img.copy(),eff,**kwargs)
            img.save(out/fname); results.append((f"FX:{eff}",out/fname))
        except Exception as e: print(f"  [!] {eff}: {e}")

    # Variations
    print("\n── 4 Variations ──")
    try:
        paths=pipe.generate_variations("magical glowing forest",n=4,
            output_dir=str(out/"variations"),width=300,height=300)
        results+=[(f"Var {i}",p) for i,p in enumerate(paths)]
    except Exception as e: print(f"  [!] Variations: {e}")

    # Interpolation
    print("\n── Prompt Interpolation (5 frames) ──")
    try:
        paths=pipe.interpolate("warm golden sunrise","cold dark night sky",
            steps=5,output_dir=str(out/"interpolation"),width=300,height=300)
        results+=[(f"Interp {i}",p) for i,p in enumerate(paths)]
    except Exception as e: print(f"  [!] Interpolation: {e}")

    # NLP analysis
    print("\n── NLP Analysis ──")
    for tp in ["a bright rainbow over a misty waterfall",
               "dark gothic castle at midnight"]:
        an=pipe.analyse(tp)
        print(f"  Prompt  : {tp}")
        print(f"  Scene   : {an.scene_type} | Colours: {[k for k,_ in an.colour_matches[:4]]}")
        print(f"  Mods    : {list(an.modifiers.keys())} | Complexity: {an.complexity():.2f}\n")

    # Custom bit-depth
    print("\n── Custom Bit-Depth ──")
    for bpc,nch,suf in [(16,3,"16bit"),(32,3,"32bit"),(8,6,"6ch")]:
        try:
            pipe.custom_bitdepth(80,80,bpc,nch,
                output=str(out/f"custom_{suf}.custimg"),
                preview=str(out/f"custom_{suf}_preview.png"))
        except Exception as e: print(f"  [!] {bpc}bpc: {e}")

    cfg.save(str(out/"tti_config.json"))

    # Model info
    info=pipe.model_info()
    print("\n── Model Info ──")
    for k,v in info.items(): print(f"  {k:20s}: {v}")

    print(f"\n{'='*65}")
    print(f"  Demo complete — {len(results)} outputs saved to '{out}/'")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p=argparse.ArgumentParser(prog="tti",description="TTI — Text-To-Image system")
    sub=p.add_subparsers(dest="command")

    g=sub.add_parser("generate"); g.add_argument("prompt")
    g.add_argument("--output","-o",default=None); g.add_argument("--width","-W",type=int,default=None)
    g.add_argument("--height","-H",type=int,default=None); g.add_argument("--seed","-s",type=int,default=None)
    g.add_argument("--format","-f",default=None,dest="fmt",choices=["png","bmp","jpeg"])

    b=sub.add_parser("batch"); b.add_argument("prompts_file")
    b.add_argument("--output-dir","-o",default="tti_output")
    b.add_argument("--width","-W",type=int,default=None); b.add_argument("--height","-H",type=int,default=None)
    b.add_argument("--seed","-s",type=int,default=None); b.add_argument("--format","-f",default="png")
    b.add_argument("--prefix","-p",default="img")

    a=sub.add_parser("art"); a.add_argument("art_type",choices=sorted(TTIPipeline._ART_TYPES))
    a.add_argument("--output","-o",default=None); a.add_argument("--width","-W",type=int,default=None)
    a.add_argument("--height","-H",type=int,default=None); a.add_argument("--seed","-s",type=int,default=None)
    a.add_argument("--n-cells",type=int,default=20); a.add_argument("--depth",type=int,default=7)
    a.add_argument("--octaves",type=int,default=4); a.add_argument("--c-real",type=float,default=-0.7)
    a.add_argument("--c-imag",type=float,default=0.27015); a.add_argument("--turns",type=int,default=6)
    a.add_argument("--freq",type=float,default=0.05); a.add_argument("--square-size",type=int,default=40)

    e=sub.add_parser("effects"); e.add_argument("effect",choices=sorted(TTIPipeline._EFFECTS))
    e.add_argument("input"); e.add_argument("--output","-o",default=None)
    e.add_argument("--radius",type=int,default=None); e.add_argument("--sigma",type=float,default=2.0)
    e.add_argument("--intensity",type=float,default=None); e.add_argument("--factor",type=float,default=1.2)
    e.add_argument("--strength",type=float,default=0.6); e.add_argument("--block-size",type=int,default=10)

    an=sub.add_parser("analyse"); an.add_argument("prompt")

    i=sub.add_parser("interpolate"); i.add_argument("prompt_a"); i.add_argument("prompt_b")
    i.add_argument("--steps",type=int,default=6); i.add_argument("--output-dir","-o",default="tti_output/interpolation")
    i.add_argument("--width","-W",type=int,default=None); i.add_argument("--height","-H",type=int,default=None)

    v=sub.add_parser("variations"); v.add_argument("prompt")
    v.add_argument("--n",type=int,default=4); v.add_argument("--output-dir","-o",default="tti_output/variations")
    v.add_argument("--width","-W",type=int,default=None); v.add_argument("--height","-H",type=int,default=None)

    an2=sub.add_parser("animate"); an2.add_argument("prompt")
    an2.add_argument("--frames",type=int,default=24); an2.add_argument("--output-dir","-o",default="tti_output/animation")
    an2.add_argument("--width","-W",type=int,default=None); an2.add_argument("--height","-H",type=int,default=None)

    st=sub.add_parser("stream"); st.add_argument("output")
    st.add_argument("--width","-W",type=int,default=2000); st.add_argument("--height","-H",type=int,default=2000)
    st.add_argument("--pattern","-p",default="gradient",choices=["gradient","checkerboard","noise"])

    cfg=sub.add_parser("config"); cfg.add_argument("--show",action="store_true")
    cfg.add_argument("--set",nargs="+",metavar="section.key=value")
    cfg.add_argument("--save",default=None); cfg.add_argument("--reset",action="store_true")

    dm=sub.add_parser("demo"); dm.add_argument("--output-dir","-o",default="tti_demo")
    return p

def _parse_set_args(args):
    for item in args:
        if "=" not in item or "." not in item: continue
        key,val=item.split("=",1); section,attr=key.split(".",1)
        try: val=int(val)
        except ValueError:
            try: val=float(val)
            except ValueError: val=True if val.lower()=="true" else (False if val.lower()=="false" else val)
        update_config(**{section:{attr:val}}); print(f"[TTI] Set {section}.{attr} = {val!r}")

def main(argv=None):
    import warnings; warnings.filterwarnings("ignore")
    parser=_build_parser(); args=parser.parse_args(argv)
    if args.command is None: parser.print_help(); return 0
    pipe=TTIPipeline(); cmd=args.command

    if cmd=="generate":
        out=args.output or f"tti_output/generated.{args.fmt or 'png'}"
        pipe.generate(args.prompt,output=out,width=args.width,height=args.height,seed=args.seed,fmt=args.fmt)
    elif cmd=="batch":
        path=Path(args.prompts_file)
        if not path.exists(): print(f"[TTI] File not found: {path}"); return 1
        prompts=[l.strip() for l in path.read_text().splitlines() if l.strip()]
        pipe.generate_batch(prompts,output_dir=args.output_dir,width=args.width,
            height=args.height,seed=args.seed,fmt=args.format,prefix=args.prefix)
    elif cmd=="art":
        out=args.output or f"tti_output/{args.art_type}.png"
        kw={"mandelbrot":{},"julia":{"c_real":args.c_real,"c_imag":args.c_imag},
            "sierpinski":{"depth":args.depth},"voronoi":{"n_cells":args.n_cells},
            "noise":{"octaves":args.octaves},"spiral":{"turns":args.turns},
            "waves":{"freq":args.freq},"checkerboard":{"square_size":args.square_size},
            "plasma":{},"gradient":{},"radial":{},"circles":{}}.get(args.art_type,{})
        pipe.art(args.art_type,output=out,width=args.width,height=args.height,seed=args.seed,**kw)
    elif cmd=="effects":
        out=args.output or f"tti_output/effect_{args.effect}.png"
        kw={"sigma":args.sigma,"factor":args.factor,"strength":args.strength,"block_size":args.block_size}
        if args.radius is not None: kw["radius"]=args.radius
        if args.intensity is not None: kw["intensity"]=args.intensity
        pipe.effect(args.effect,args.input,output=out,**kw)
    elif cmd=="analyse":
        an=pipe.analyse(args.prompt)
        print(f"\nPrompt    : {an.raw_prompt}\nScene     : {an.scene_type}")
        print(f"Colours   : {[k for k,_ in an.colour_matches]}\nNouns     : {an.nouns}")
        print(f"Adjectives: {an.adjectives}\nModifiers : {list(an.modifiers.keys())}")
        print(f"Tokens    : {an.filtered_tokens}\nComplexity: {an.complexity():.2f}")
    elif cmd=="interpolate":
        pipe.interpolate(args.prompt_a,args.prompt_b,steps=args.steps,
            output_dir=args.output_dir,width=args.width,height=args.height)
    elif cmd=="variations":
        pipe.generate_variations(args.prompt,n=args.n,output_dir=args.output_dir,
            width=args.width,height=args.height)
    elif cmd=="animate":
        pipe.animate(args.prompt,n_frames=args.frames,output_dir=args.output_dir,
            width=args.width,height=args.height)
    elif cmd=="stream":
        pipe.stream_large(args.output,args.width,args.height,args.pattern)
    elif cmd=="config":
        if args.reset: reset_config(); print("[TTI] Config reset.")
        if args.set: _parse_set_args(args.set)
        if args.show: pipe.show_config()
        if args.save: pipe.save_config(args.save)
    elif cmd=="demo":
        demo(output_dir=args.output_dir)
    return 0

if __name__=="__main__":
    sys.exit(main())