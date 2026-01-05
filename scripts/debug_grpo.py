import importlib
import inspect


def inspect_grpo_learner():
    try:
        from tunix.rl.batch_job import GRPOLearner

        print("Successfully imported GRPOLearner from tunix.rl.batch_job")
    except ImportError:
        try:
            from tunix.rl.rl_cluster import GRPOLearner

            print("Successfully imported GRPOLearner from tunix.rl.rl_cluster")
        except ImportError:
            print(
                "Could not import GRPOLearner from batch_job or rl_cluster. Searching tunix package..."
            )
            # Fallback search
            import tunix

            found = False
            for name, obj in inspect.getmembers(tunix):
                if inspect.ismodule(obj):
                    for sub_name, sub_obj in inspect.getmembers(obj):
                        if sub_name == "GRPOLearner":
                            print(f"Found GRPOLearner in {name}")
                            GRPOLearner = sub_obj
                            found = True
                            break
            if not found:
                print("GRPOLearner not found in top-level modules.")
                return

    print("\nAttributes of GRPOLearner:")
    for attr in dir(GRPOLearner):
        if not attr.startswith("__"):
            print(f" - {attr}")

    # Also check if we can instantiate it purely for introspection if that helps,
    # but printing dir(class) is usually enough to see method names.


if __name__ == "__main__":
    inspect_grpo_learner()
