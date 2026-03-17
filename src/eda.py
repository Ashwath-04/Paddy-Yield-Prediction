import matplotlib.pyplot as plt
import seaborn as sns

def basic_info(df):
    print("\nDataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())


def plot_target_distribution(df, target):
    plt.figure()
    sns.histplot(df[target], kde=True)
    plt.title("Target Distribution")
    plt.savefig("outputs/plots/target_distribution.png")


def correlation_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", vmax=1, vmin=-1)
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/plots/correlation.png")