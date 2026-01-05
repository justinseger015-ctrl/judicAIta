#!/usr/bin/env python3
"""
Tunix Module Inspection Script
==============================
Run this script in Google Colab to inspect the tunix.rl.rl_cluster module
and identify the correct class names for RolloutConfig, RLTrainingConfig, and ClusterConfig.

Usage (in Colab cell):
    !python scripts/inspect_tunix_modules.py

Or copy/paste contents directly into a Colab cell.
"""

import sys
import inspect


def inspect_class_details(cls, prefix=""):
    """Print detailed class information including init signature."""
    print(f"{prefix}Class: {cls.__name__}")

    # Get __init__ signature if available
    try:
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.items())
        if params and params[0][0] == "self":
            params = params[1:]  # Skip 'self'
        if params:
            print(f"{prefix}  __init__ parameters:")
            for pname, param in params[:10]:  # Limit to first 10
                default = param.default
                if default == inspect.Parameter.empty:
                    print(f"{prefix}    - {pname} (required)")
                else:
                    print(f"{prefix}    - {pname} = {default}")
            if len(params) > 10:
                print(f"{prefix}    ... and {len(params) - 10} more")
    except Exception as e:
        print(f"{prefix}  (Could not inspect __init__: {e})")

    # Get class docstring
    if cls.__doc__:
        doc_line = cls.__doc__.strip().split("\n")[0][:80]
        print(f"{prefix}  Doc: {doc_line}")


def inspect_module(module_path: str, indent: int = 0, show_details: bool = False) -> None:
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
                if show_details and is_config:
                    inspect_class_details(cls, prefix + "      ")

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
    print("🔍 TUNIX RL_CLUSTER MODULE INSPECTION")
    print("=" * 70)
    print()
    print("Looking for: RolloutConfig, RLTrainingConfig, ClusterConfig")
    print()

    # Primary focus: rl_cluster module
    print("=" * 70)
    print("📦 INSPECTING tunix.rl.rl_cluster (PRIMARY)")
    print("=" * 70)
    inspect_module("tunix.rl.rl_cluster", show_details=True)

    # Also check base_rollout module for rollout configs
    print("\n" + "=" * 70)
    print("📦 INSPECTING tunix.rl.rollout.base_rollout")
    print("=" * 70)
    inspect_module("tunix.rl.rollout.base_rollout", show_details=True)

    # Check tunix.rl module directly
    print("\n" + "=" * 70)
    print("📦 INSPECTING tunix.rl")
    print("=" * 70)
    inspect_module("tunix.rl", show_details=True)

    # Check grpo learner for training configs
    print("\n" + "=" * 70)
    print("📦 INSPECTING tunix.rl.grpo.grpo_learner")
    print("=" * 70)
    inspect_module("tunix.rl.grpo.grpo_learner", show_details=True)

    # Additional modules for context
    print("\n" + "=" * 70)
    print("📦 OTHER MODULES FOR CONTEXT")
    print("=" * 70)

    other_modules = [
        "tunix.rl.grpo",
        "tunix.rl.rollout",
        "tunix.models.gemma3.model",
    ]

    for module_path in other_modules:
        print()
        inspect_module(module_path, show_details=False)
        print("-" * 50)

    # Deep search for config classes across all RL modules
    print("\n" + "=" * 70)
    print("🔎 DEEP SEARCH FOR CONFIG CLASSES IN RL MODULES")
    print("=" * 70)

    rl_modules = [
        "tunix.rl",
        "tunix.rl.rl_cluster",
        "tunix.rl.grpo",
        "tunix.rl.grpo.grpo_learner",
        "tunix.rl.rollout",
        "tunix.rl.rollout.base_rollout",
    ]

    config_candidates = []

    for module_path in rl_modules:
        try:
            module = __import__(module_path, fromlist=[""])
            for attr in dir(module):
                if not attr.startswith("_"):
                    try:
                        obj = getattr(module, attr)
                        if isinstance(obj, type):
                            name_lower = attr.lower()
                            keywords = ["config", "rollout", "cluster", "training", "params"]
                            if any(kw in name_lower for kw in keywords):
                                config_candidates.append((module_path, attr, obj))
                    except Exception:
                        pass
        except ImportError:
            pass

    if config_candidates:
        print("\n⭐ Found relevant classes:")
        for mod, name, cls in config_candidates:
            print(f"\n   from {mod} import {name}")
            inspect_class_details(cls, "      ")
    else:
        print("\n⚠️ No config classes found in tunix.rl modules")

    # Provide suggested fix
    print("\n" + "=" * 70)
    print("💡 SUGGESTED FIX FOR RolloutConfig AttributeError")
    print("=" * 70)
    print("""
Based on tunix 0.1.5, the rollout configuration may be:

OPTION 1: Check if RolloutConfig is in base_rollout module:
    from tunix.rl.rollout.base_rollout import RolloutConfig

OPTION 2: RolloutConfig may be nested inside ClusterConfig:
    # Pass rollout params directly to ClusterConfig or RLCluster

OPTION 3: The params may be passed directly to RLCluster:
    rl_cluster = RLCluster(
        actor_model=actor_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        max_tokens_to_generate=512,
        max_prompt_length=1024,
        temperature=0.7,
        ...
    )

Run this script in Colab to see the actual available classes!
""")


if __name__ == "__main__":
    main()
