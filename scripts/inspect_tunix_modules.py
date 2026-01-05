#!/usr/bin/env python3
"""
Tunix Module Inspection Script
==============================
Run this script in Google Colab to inspect the tunix.models.gemma package
and identify the correct class names for configuration and model wrappers.

Usage (in Colab cell):
    !python scripts/inspect_tunix_modules.py
    
Or copy/paste contents directly into a Colab cell.
"""

import sys


def inspect_module(module_path: str, indent: int = 0) -> None:
    """Recursively inspect a module and print its contents."""
    prefix = "  " * indent
    try:
        module = __import__(module_path, fromlist=[""])
        print(f"{prefix}📦 {module_path}")
        
        # Get all public attributes
        attrs = [a for a in dir(module) if not a.startswith("_")]
        
        # Categorize attributes
        classes = []
        functions = []
        submodules = []
        other = []
        
        for attr in attrs:
            try:
                obj = getattr(module, attr)
                if isinstance(obj, type):
                    classes.append((attr, obj))
                elif callable(obj):
                    functions.append(attr)
                elif hasattr(obj, "__path__") or hasattr(obj, "__file__"):
                    submodules.append(attr)
                else:
                    other.append(attr)
            except Exception:
                other.append(attr)
        
        # Print classes (most important for finding Config classes)
        if classes:
            print(f"{prefix}  📋 Classes:")
            for name, cls in classes:
                # Check if it looks like a config class
                is_config = "config" in name.lower()
                marker = "⭐" if is_config else "  "
                bases = ", ".join(b.__name__ for b in cls.__bases__[:3])
                print(f"{prefix}    {marker} {name} (bases: {bases})")
        
        # Print functions
        if functions:
            print(f"{prefix}  🔧 Functions:")
            for name in functions[:15]:  # Limit to first 15
                print(f"{prefix}       {name}")
            if len(functions) > 15:
                print(f"{prefix}       ... and {len(functions) - 15} more")
        
        # Print submodules
        if submodules:
            print(f"{prefix}  📁 Submodules:")
            for name in submodules[:10]:
                print(f"{prefix}       {name}")
        
    except ImportError as e:
        print(f"{prefix}❌ Failed to import {module_path}: {e}")
    except Exception as e:
        print(f"{prefix}⚠️ Error inspecting {module_path}: {e}")


def main():
    print("=" * 70)
    print("🔍 TUNIX MODULE INSPECTION")
    print("=" * 70)
    print()
    
    # Module paths to inspect
    modules_to_check = [
        # Gemma modules
        "tunix",
        "tunix.models",
        "tunix.models.gemma",
        "tunix.models.gemma.model",
        "tunix.models.gemma3",
        "tunix.models.gemma3.model",
        # RL modules
        "tunix.rl",
        "tunix.rl.grpo",
        "tunix.rl.grpo.grpo_learner",
        # Generate modules
        "tunix.generate",
        "tunix.generate.tokenizer_adapter",
    ]
    
    for module_path in modules_to_check:
        print()
        inspect_module(module_path)
        print("-" * 50)
    
    print()
    print("=" * 70)
    print("🔎 SEARCHING FOR CONFIG-LIKE CLASSES")
    print("=" * 70)
    
    # Deep search for config classes
    config_candidates = []
    
    for module_path in modules_to_check:
        try:
            module = __import__(module_path, fromlist=[""])
            for attr in dir(module):
                if not attr.startswith("_"):
                    try:
                        obj = getattr(module, attr)
                        if isinstance(obj, type):
                            name_lower = attr.lower()
                            if any(kw in name_lower for kw in ["config", "params", "args", "options", "settings"]):
                                config_candidates.append((module_path, attr, obj))
                    except Exception:
                        pass
        except ImportError:
            pass
    
    if config_candidates:
        print("\n⭐ Found config-like classes:")
        for mod, name, cls in config_candidates:
            print(f"   from {mod} import {name}")
            # Try to show docstring or signature
            if cls.__doc__:
                doc_line = cls.__doc__.strip().split("\n")[0][:60]
                print(f"      → {doc_line}...")
    else:
        print("\n⚠️ No obvious config classes found in tunix modules")
    
    print()
    print("=" * 70)
    print("💡 SUGGESTED: Try using transformers.GemmaConfig instead")
    print("=" * 70)
    print("""
If tunix doesn't expose a GemmaConfig, you can use HuggingFace's:

    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_path)

Or specifically:

    from transformers import Gemma3Config  # or GemmaConfig for older versions
    model_config = Gemma3Config.from_pretrained(model_path)

The model_config may not even be needed if gemma_lib.GemmaForCausalLM.from_pretrained()
automatically loads the config internally.
""")


if __name__ == "__main__":
    main()
