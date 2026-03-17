from sklearn.metrics import r2_score, mean_absolute_error

def evaluate(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{name} R2:", r2)
    print(f"{name} MAE:", mae)

    # Save report
    with open(f"outputs/reports/{name}_metrics.txt", "w") as f:
        f.write(f"R2 Score: {r2}\n")
        f.write(f"MAE: {mae}\n")