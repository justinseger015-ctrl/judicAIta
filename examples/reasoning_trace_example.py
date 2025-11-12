"""
Example: Generate reasoning trace for legal analysis.

This script demonstrates how to use the reasoning trace generator
to create explainable step-by-step legal analysis.
"""

import asyncio

from judicaita.reasoning_trace import ReasoningTraceGenerator


async def main():
    """Main example function."""
    print("=" * 60)
    print("Judicaita Reasoning Trace Example")
    print("=" * 60)

    # Initialize the generator
    generator = ReasoningTraceGenerator()
    await generator.initialize()

    # Sample legal query
    query = """
    Does the plaintiff have standing to sue under the Equal Protection Clause
    based on discriminatory school district policies?
    """

    # Sample context
    context = """
    The plaintiff is a parent of a student in the school district. The district
    has implemented policies that result in unequal allocation of resources
    between schools in predominantly white neighborhoods versus those in
    predominantly minority neighborhoods.
    """

    # Sample relevant citations
    citations = [
        "Brown v. Board of Education, 347 U.S. 483 (1954)",
        "Plyler v. Doe, 457 U.S. 202 (1982)",
        "42 U.S.C. § 2000d",
    ]

    print("\n1. Generating reasoning trace...")
    print("-" * 60)
    print(f"Query: {query.strip()}")

    # Generate trace
    trace = await generator.generate_trace(
        query=query.strip(),
        context=context.strip(),
        citations=citations,
    )

    print(f"\n2. Trace generated successfully!")
    print(f"   Trace ID: {trace.trace_id}")
    print(f"   Overall Confidence: {trace.overall_confidence:.2%}")
    print(f"   Steps: {len(trace.steps)}")

    print("\n3. Reasoning steps:")
    print("-" * 60)

    for i, step in enumerate(trace.steps, 1):
        print(f"\nStep {i}: {step.step_type.value.upper()}")
        print(f"  Description: {step.description}")
        print(f"  Confidence: {step.confidence_score:.2%}")
        if step.sources:
            print(f"  Sources: {', '.join(step.sources[:3])}")

    print("\n4. Final conclusion:")
    print("-" * 60)
    print(trace.final_conclusion)

    print("\n5. Citations used:")
    print("-" * 60)
    for citation in trace.citations_used:
        print(f"  - {citation}")

    print("\n6. Exporting to markdown...")
    markdown = trace.to_markdown()
    output_path = "reasoning_trace_example.md"
    with open(output_path, "w") as f:
        f.write(markdown)
    print(f"   Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
