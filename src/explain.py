"""Natural-language explanations for the agent's decisions.

* TemplateExplainer  - always available, no internet.
* WatsonxExplainer   - OPTIONAL: IBM watsonx.ai (Granite model). Set environment variables
                       WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL (and optionally WATSONX_MODEL_ID).
                       Falls back to the template if anything fails.
"""
import os

PHASE_NAME = {0: "North-South", 1: "East-West"}


class TemplateExplainer:
    def explain(self, ctx):
        new = PHASE_NAME[1 - ctx["phase"]]
        msg = (f"Switched green to {new}: {ctx['red_q']:.0f} vehicles waiting on the red side vs "
               f"{ctx['green_q']:.0f} on the green side.")
        if ctx.get("forecast_level"):
            msg += f" Forecast demand is {ctx['forecast_level']}."
        return msg


class WatsonxExplainer:
    def __init__(self, max_calls=5):
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        creds = Credentials(url=os.environ["WATSONX_URL"], api_key=os.environ["WATSONX_API_KEY"])
        self.model = ModelInference(model_id=os.environ.get("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct"),
                                    credentials=creds, project_id=os.environ["WATSONX_PROJECT_ID"],
                                    params={"max_new_tokens": 60, "temperature": 0.2})
        self.fallback, self.max_calls, self.calls = TemplateExplainer(), max_calls, 0

    def explain(self, ctx):
        base = self.fallback.explain(ctx)
        if self.calls >= self.max_calls:
            return base
        self.calls += 1
        try:
            prompt = ("You are a traffic control assistant. In one short sentence for a city operator, "
                      f"explain this signal decision: {base}")
            return self.model.generate_text(prompt=prompt).strip() or base
        except Exception:
            return base


def get_explainer(use_watsonx=False):
    if use_watsonx:
        try:
            return WatsonxExplainer()
        except Exception as e:
            print("watsonx unavailable, using template explanations:", e)
    return TemplateExplainer()
