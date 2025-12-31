import time
import random
import gradio as gr
from typing import List
from app.services.ml_service import model_service

NUMERICAL_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

def get_default_value(col):
    if col == "Time":
        return 1000.0
    if col == "Amount":
        return 50.0
    return 0.0

DEFAULT_VALUES = [get_default_value(c) for c in NUMERICAL_COLUMNS]

def build_feature_dict(values: List[float]):
    return dict(zip(NUMERICAL_COLUMNS, values))

def safe_format_probability(p):
    try:
        return f"{float(p):.4f}"
    except Exception:
        return "N/A"

def generate_random_request_inputs(deterministic: bool = False) -> List[float]:
    if deterministic:
        random.seed(42)

    values = []
    values.append(float(random.randint(0, 172800)))  # Time
    for _ in range(28):
        values.append(random.normalvariate(0, 1.5))
    values.append(round(random.uniform(0.01, 5000.00), 2))
    return values

def reset_inputs():
    return DEFAULT_VALUES.copy()

def predict_generator(*args):
    if len(args) != len(NUMERICAL_COLUMNS):
        yield "**Input error**: wrong number of fields."
        return

    yield "⏳ Running model inference…"

    features = build_feature_dict(args)

    try:
        start = time.time()
        data = model_service.predict(features)
        latency = time.time() - start

        prob = safe_format_probability(data.get("fraud_probability"))
        version = data.get("model_version", "unknown")

        risk_label = ""
        try:
            p = float(prob)
            if p >= 0.7:
                risk_label = "🔴 High risk"
            elif p >= 0.4:
                risk_label = "🟠 Medium risk"
            else:
                risk_label = "🟢 Low risk"
        except:
            pass

        md = f"""
## Prediction Result

**Fraud Probability**: **{prob}** {risk_label}

**Model Version**: `{version}`  
**Inference Time**: `{latency:.3f} sec`
"""
        yield md

    except Exception as e:
        yield f"❌ Prediction error: {e}"


custom_theme = gr.themes.Soft().set(
    button_primary_background_fill_hover='*primary_600',
)

with gr.Blocks(title="Fraud Detection Service Demo") as demo:

    gr.Markdown("<h2 style='text-align:center;'>Fraud Prediction Service</h2>")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Column(variant="panel"):
                gr.Markdown("### Transaction Inputs")

                with gr.Row():
                    predict_btn = gr.Button("🚀 Predict", variant="primary")
                    random_btn = gr.Button("🎲 Random")
                    reset_btn = gr.Button("🔄 Reset")
                deterministic_checkbox = gr.Checkbox(label="Deterministic Random (seed=42)", value=False)

                feature_inputs = []

                with gr.Row():
                    with gr.Column():
                        for col in NUMERICAL_COLUMNS[:15]:
                            if col == "Time":
                                comp = gr.Number(label=col, value=1000.0, minimum=0)
                            elif col == "Amount":
                                comp = gr.Number(label=col, value=50.0, minimum=0.0, step=0.01)
                            else:
                                comp = gr.Number(label=col, value=0.0, step=0.0001)
                            feature_inputs.append(comp)

                    with gr.Column():
                        for col in NUMERICAL_COLUMNS[15:]:
                            if col == "Time":
                                comp = gr.Number(label=col, value=1000.0, minimum=0)
                            elif col == "Amount":
                                comp = gr.Number(label=col, value=50.0, minimum=0.0, step=0.01)
                            else:
                                comp = gr.Number(label=col, value=0.0, step=0.0001)
                            feature_inputs.append(comp)

        with gr.Column(scale=1):
            with gr.Column(variant="panel"):
                gr.Markdown("### Prediction Output")
                output_md = gr.Markdown("Fill the inputs and click Predict.")

    random_btn.click(
        fn=generate_random_request_inputs,
        inputs=[deterministic_checkbox],
        outputs=feature_inputs,
        queue=False
    )

    reset_btn.click(
        fn=reset_inputs,
        inputs=None,
        outputs=feature_inputs,
        queue=False
    )

    predict_btn.click(
        fn=predict_generator,
        inputs=feature_inputs,
        outputs=output_md,
        queue=True
    )

gr_app = demo