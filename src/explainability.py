import shap
import matplotlib.pyplot as plt

def explain_model(model, X, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("outputs/plots/shap_beeswarm.png")

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.savefig("outputs/plots/shap_bar.png")