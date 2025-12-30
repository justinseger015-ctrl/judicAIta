"""
Command-line interface for Judicaita.
"""

from pathlib import Path

import typer
from rich.console import Console

from judicaita import __version__

app = typer.Typer(
    name="judicaita",
    help="Judicaita: Explainable Legal AI Assistant",
    add_completion=False,
)

console = Console()


@app.command()
def version() -> None:
    """Display version information."""
    console.print(f"Judicaita version {__version__}", style="bold green")


@app.command()
def process_document(
    file_path: Path = typer.Argument(..., help="Path to document to process"),
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
    generate_summary: bool = typer.Option(
        True, "--summary/--no-summary", help="Generate plain-English summary"
    ),
    extract_citations: bool = typer.Option(
        True, "--citations/--no-citations", help="Extract and map citations"
    ),
    create_trace: bool = typer.Option(True, "--trace/--no-trace", help="Create reasoning trace"),
) -> None:
    """
    Process a legal document and generate analysis.

    Args:
        file_path: Path to the document
        output_dir: Output directory for results
        generate_summary: Generate plain-English summary
        extract_citations: Extract and map citations
        create_trace: Create reasoning trace
    """
    console.print(f"Processing document: {file_path}", style="bold")

    # TODO: Implement document processing pipeline
    console.print("[yellow]Document processing is pending model integration.[/yellow]")


@app.command()
def analyze_query(
    query: str = typer.Argument(..., help="Legal query to analyze"),
    context_file: Path | None = typer.Option(None, "--context", "-c", help="Context document"),
    output_format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format (markdown, json)"
    ),
) -> None:
    """
    Analyze a legal query with reasoning trace.

    Args:
        query: Legal query to analyze
        context_file: Optional context document
        output_format: Output format
    """
    console.print(f"Analyzing query: {query}", style="bold")

    # TODO: Implement query analysis
    console.print("[yellow]Query analysis is pending model integration.[/yellow]")


@app.command()
def audit_report(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to include"),
    output_file: Path | None = typer.Option(None, "--output", "-o", help="Output file for report"),
) -> None:
    """
    Generate compliance audit report.

    Args:
        days: Number of days to include in report
        output_file: Output file for report
    """
    console.print(f"Generating audit report for last {days} days", style="bold")

    # TODO: Implement audit report generation
    console.print("[yellow]Audit report generation pending implementation.[/yellow]")


@app.command()
def validate_citation(
    citation: str = typer.Argument(..., help="Citation to validate"),
) -> None:
    """
    Validate a legal citation.

    Args:
        citation: Citation string to validate
    """
    console.print(f"Validating citation: {citation}", style="bold")

    # TODO: Implement citation validation
    console.print("[yellow]Citation validation pending implementation.[/yellow]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """
    Start the Judicaita API server.

    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload for development
    """
    import uvicorn

    console.print(f"Starting Judicaita API server on {host}:{port}", style="bold green")

    try:
        uvicorn.run(
            "judicaita.api:app",
            host=host,
            port=port,
            reload=reload,
        )
    except ImportError:
        console.print("[red]API module not yet implemented. Check back soon![/red]")


@app.command()
def train_grpo(
    output_dir: Path = typer.Option(
        "./checkpoints/grpo", "--output-dir", "-o", help="Output directory for checkpoints"
    ),
    base_model: str = typer.Option(
        "google/gemma-2-2b-it", "--base-model", "-m", help="Base model to fine-tune"
    ),
    num_epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate", "-lr", help="Learning rate"),
    use_lora: bool = typer.Option(True, "--lora/--no-lora", help="Use LoRA adaptation"),
    generate_cot: bool = typer.Option(
        False, "--generate-cot/--no-generate-cot", help="Generate synthetic CoT traces"
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Maximum training samples (for testing)"
    ),
    # Phase 2: Validation mode options
    max_steps: int | None = typer.Option(
        None, "--max-steps", help="Maximum training steps (for validation runs, e.g., 50)"
    ),
    validation_mode: bool = typer.Option(
        False,
        "--validation-mode/--no-validation-mode",
        help="Enable validation mode with profiling (sets max_steps=50, num_rollouts=2)",
    ),
    memory_threshold: float = typer.Option(
        12.0, "--memory-threshold", help="Memory threshold in GB for validation warnings"
    ),
    time_limit: float = typer.Option(
        8.5, "--time-limit", help="Maximum allowed training time in hours (Kaggle limit)"
    ),
) -> None:
    """
    Train a model using GRPO on legal reasoning tasks.

    Use --validation-mode for Phase 2 validation runs with memory profiling
    and time estimation. This automatically sets max_steps=50 and num_rollouts=2.

    Args:
        output_dir: Output directory for checkpoints
        base_model: Base model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_lora: Use LoRA for parameter-efficient training
        generate_cot: Generate synthetic CoT traces
        max_samples: Maximum training samples
        max_steps: Maximum training steps for validation
        validation_mode: Enable validation mode with profiling
        memory_threshold: Memory threshold in GB
        time_limit: Maximum training time in hours
    """
    from judicaita.training import GRPOTrainer, TrainingConfig, create_training_dataset

    console.print("[bold]Starting GRPO training pipeline...[/bold]")

    # Apply validation mode defaults
    if validation_mode:
        console.print("[yellow]🔬 Validation mode enabled[/yellow]")
        if max_steps is None:
            max_steps = 50
            console.print(f"   Setting max_steps={max_steps}")
        num_rollouts = 2
        console.print(f"   Setting num_rollouts={num_rollouts}")
    else:
        num_rollouts = 4

    try:
        # Create training dataset
        console.print("Loading and preparing datasets...")
        legalbench_data, pile_of_law_data = create_training_dataset(generate_cot=generate_cot)

        # Limit samples if specified
        if max_samples is not None:
            legalbench_data = legalbench_data.select(range(min(max_samples, len(legalbench_data))))

        console.print(
            f"Loaded {len(legalbench_data)} LegalBench samples, "
            f"{len(pile_of_law_data)} Pile-of-Law samples"
        )

        # Create training config
        config = TrainingConfig(
            base_model=base_model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_lora=use_lora,
            checkpoint_dir=output_dir,
            num_rollouts=num_rollouts,
            max_steps=max_steps,
            validation_mode=validation_mode,
            memory_threshold_gb=memory_threshold,
            time_limit_hours=time_limit,
        )

        # Initialize trainer
        console.print("Initializing GRPO trainer...")
        trainer = GRPOTrainer(config=config, train_dataset=legalbench_data)
        trainer.initialize()

        # Start training
        console.print("[bold green]Starting training...[/bold green]")
        metrics = trainer.train()

        # Check validation results
        if validation_mode or max_steps:
            validation_report = metrics.get("validation_report", {})
            if validation_report.get("ready_for_phase_3"):
                console.print(
                    "[bold green]✅ Phase 2 validation passed - Ready for Phase 3![/bold green]"
                )
            else:
                console.print(
                    "[bold yellow]⚠️ Phase 2 validation failed - Review report above[/bold yellow]"
                )

        # Always show a final metrics summary for all runs
        console.print(f"Final metrics: {metrics}")
        console.print("[bold green]Training completed successfully![/bold green]")
        console.print(f"Checkpoints saved to: {output_dir}")

    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command()
def evaluate_model(
    checkpoint_path: Path = typer.Argument(..., help="Path to model checkpoint"),
    output_file: Path | None = typer.Option(None, "--output", "-o", help="Output file for results"),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Maximum samples to evaluate"
    ),
) -> None:
    """
    Evaluate a trained model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        output_file: Output file for results
        max_samples: Maximum samples to evaluate
    """
    from judicaita.training import create_training_dataset
    from judicaita.training.evaluation import evaluate_checkpoint

    console.print(f"[bold]Evaluating model: {checkpoint_path}[/bold]")

    try:
        # Load evaluation dataset
        console.print("Loading evaluation dataset...")
        legalbench_data, _ = create_training_dataset(generate_cot=False)

        # Split for evaluation (use last 20%)
        eval_size = int(0.2 * len(legalbench_data))
        eval_dataset = legalbench_data.select(
            range(len(legalbench_data) - eval_size, len(legalbench_data))
        )

        console.print(f"Evaluating on {len(eval_dataset)} samples")

        # Run evaluation
        evaluate_checkpoint(
            str(checkpoint_path),
            eval_dataset,
            max_samples=max_samples,
            output_file=str(output_file) if output_file else None,
        )

        console.print("[bold green]Evaluation completed![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise typer.Exit(1) from e


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
