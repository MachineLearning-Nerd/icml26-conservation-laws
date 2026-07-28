import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Conservation laws: an evidence-first reproduction

    | Claim | Formal evidence | Verdict |
    | --- | ---: | --- |
    | GELU / SiLU completeness | exact ranks 2,4,…,16 | **VERIFIED** |
    | SwiGLU | relative rate `3.91e-16` | **VERIFIED** |
    | MHA + RoPE | RoPE rate `6.66e-16` | **VERIFIED** |
    | MoE variants | sigmoid derivative `9/250` | **FALSIFIED** |
    | Four-dataset Figure 2 | 4 routes, exact run unavailable | **BLOCKED** |

    These values are embedded from immutable formal runs. Viewing this
    notebook does not rerun expensive experiments.
    """)
    return


@app.cell
def _(mo):
    exact_derivative = 9 / 250
    mo.md(
        rf"""
        ## The central counterexample

        A normalized-sigmoid MoE is supposed to share softmax's gate-row-sum
        law. For the exact two-expert construction,

        \[
        \frac{{d}}{{dt}}(W_{{1,1}}+W_{{2,1}})=\frac{{9}}{{250}}
        ={exact_derivative:.6f},
        \]

        while the matched softmax value is exactly zero. A nonzero
        instantaneous derivative is enough because the theorem quantifies over
        every dataset and initialization.
        """
    )
    return (exact_derivative,)


@app.cell
def _(mo):
    rate = mo.ui.slider(
        start=0.00001,
        stop=0.01,
        step=0.00001,
        value=0.001,
        label="Illustrative Euler step size",
    )
    rate
    return (rate,)


@app.cell
def _(exact_derivative, mo, rate):
    predicted_first_order_drift = exact_derivative * rate.value
    mo.md(
        rf"""
        The counterexample's first-order one-step drift at this illustrative
        step size is **{predicted_first_order_drift:.6g}**. This bounded
        interaction is explanatory only; it is not part of the formal
        reproduction evidence.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the other laws survive

    - **SwiGLU:** `A[:,i] → exp(t)A[:,i]` and
      `C[i,:] → exp(-t)C[i,:]` cancel exactly.
    - **MHA:** `Q→QE, K→KE^{-T}` leaves scores unchanged, and the
      analogous V/O action leaves values unchanged.
    - **RoPE:** a fixed rotation lies between Q/K blocks with exponents
      `+1` and `−1`, so their product is unchanged at every position pair.

    Differentiating each exact model symmetry and substituting Euclidean
    gradient flow yields the displayed invariant.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why Figure 2 is BLOCKED

    The exact paper scope is at least 120 individual configurations
    (four datasets × three rates × ten seeds). WikiText-103 alone entails
    450,000 optimizer steps and 5.53 billion token positions. The
    CPU-calibrated lower bound is about 732,000 seconds while omitting
    backward propagation, attention, experts, and three entire datasets.

    A proxy would be easy to run, but it would not test the claim. The
    reproducible conclusion is therefore **BLOCKED**, with the precise
    missing capability recorded.
    """)
    return


if __name__ == "__main__":
    app.run()
