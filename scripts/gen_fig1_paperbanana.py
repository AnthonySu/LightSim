"""Generate Figure 1 (CTM explanation) using PaperBanana with OpenRouter."""

import asyncio
from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
from paperbanana.core.config import Settings

SOURCE_CONTEXT = """
Cell Transmission Model (CTM) for Traffic Signal Control

LightSim's traffic dynamics are governed by the Cell Transmission Model with a triangular fundamental diagram.
Each road link is discretized into cells of length Δx = v_f · Δt, where v_f is the free-flow speed and Δt is the simulation time step.

The state of each cell i at time t is characterized by its density k_i(t) (vehicles per meter per lane).
The flow between adjacent cells is determined by the sending and receiving functions:
  S_i(k) = min(v_f · k_i, Q) · ℓ_i   (sending flow: max flow a cell can emit)
  R_i(k) = min(Q, w · (k_j - k_i)) · ℓ_i   (receiving flow: max flow a cell can accept)

where Q is the per-lane capacity (veh/s), w is the backward wave speed, k_j is the jam density, and ℓ_i is the number of lanes.

The actual intra-link flow from cell i to its downstream neighbor i+1 is:
  q_{i→i+1} = min(S_i, R_{i+1}) · Δt

At signalized intersections, movements connect the last cell of an incoming link to the first cell of an outgoing link.
Each movement m has a turn ratio β_m and saturation rate s_m.
The intersection flow is modulated by a binary signal mask σ_m ∈ {0, 1}:
  q_m = min(β_m · S_from · σ_m, s_m, R_to) · Δt

The figure should have three horizontal panels side by side:

Panel (a) "Cell Discretization": Shows a road link divided into 5 cells in a horizontal row.
Each cell is a colored rectangle (green→yellow→orange gradient based on density).
Cell labels below: "Cell 1" through "Cell 5". Inside each cell: density value k=0.02, 0.04, 0.08, 0.12, 0.05.
Small arrows between cells show flow direction (left to right). Below: "Link (road segment)".
Above: equation Δx = v_f · Δt.

Panel (b) "Sending & Receiving Flows": Shows two adjacent cells (Cell i and Cell i+1) with a large arrow between them.
Above Cell i: "S_i = min(v_f·k, Q)·ℓ" (sending flow formula).
Above Cell i+1: "R_{i+1} = min(Q, w·(k_j−k))·ℓ" (receiving flow formula).
Between cells: "q = min(S_i, R_{i+1})·Δt" flow equation on the arrow.
Below: "Flow = min(supply, demand)".

Panel (c) "Signal-Controlled Intersection": A 4-leg intersection with a central signal node.
North-South direction has GREEN signal (arrows flowing). East-West has RED signal (dashed/blocked arrows).
Central node labeled "P0" (phase 0 active). Small label "σ_m ∈ {0,1}" near the signal.
NS approaches show solid green arrows (permitted flow). EW approaches show dashed red arrows (blocked).
"""

CAPTION = """Three-panel horizontal figure explaining Cell Transmission Model mechanics for a traffic signal control paper.
Panel (a): Cell discretization of a road link into 5 cells with density color gradient (green=low, red=high).
Panel (b): Sending and receiving flow computation between two adjacent cells with mathematical formulas rendered clearly ABOVE the cells (not overlapping).
Panel (c): Signal-controlled 4-leg intersection showing green (NS) and red (EW) phases.
CRITICAL: All mathematical formulas must be properly rendered (not raw LaTeX), clearly positioned with NO overlapping of text and diagram elements. Use clean academic style with adequate spacing."""

async def main():
    settings = Settings(
        vlm_provider="openrouter",
        vlm_model="google/gemini-2.5-flash",
        image_provider="openrouter_imagen",
        image_model="google/gemini-2.5-flash-image",
        refinement_iterations=3,
        output_resolution="2k",
        output_dir="outputs/fig1_ctm",
        save_iterations=True,
    )
    pipeline = PaperBananaPipeline(settings=settings)

    result = await pipeline.generate(
        GenerationInput(
            source_context=SOURCE_CONTEXT,
            communicative_intent=CAPTION,
            diagram_type=DiagramType.METHODOLOGY,
        )
    )
    print(f"Figure saved to: {result.image_path}")
    print(f"Score: {result.score}")

if __name__ == "__main__":
    asyncio.run(main())
