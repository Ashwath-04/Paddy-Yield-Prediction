import optuna.visualization as vis

def save_optuna_plots(study, prefix):
    fig1 = vis.plot_optimization_history(study)
    fig2 = vis.plot_param_importances(study)

    fig1.write_image(f"outputs/plots/{prefix}_history.png")
    fig2.write_image(f"outputs/plots/{prefix}_importance.png")