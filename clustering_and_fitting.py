"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    """Plot a relational plot between Female and Male Life Expectancy."""
    fig, ax = plt.subplots()
    sns.scatterplot(x="Sum of Males  Life Expectancy",
                    y="Sum of Females  Life Expectancy",
                    data=df, ax=ax)
    ax.set_title("Male vs Female Life Expectancy")
    plt.savefig('relational_plot.png')
    plt.show()
    return


def plot_categorical_plot(df):
    """Plot a categorical plot (boxplot) for Life Expectancy of both sexes."""
    fig, ax = plt.subplots()
    sns.boxplot(data=df[["Sum of Males  Life Expectancy",
                         "Sum of Females  Life Expectancy",
                         "Sum of Life Expectancy  (both sexes)"]], ax=ax)
    ax.set_title("Distribution of Life Expectancy")
    plt.savefig('categorical_plot.png')
    plt.show()
    return


def plot_statistical_plot(df):
    """Plot a histogram of life expectancy for both sexes."""
    fig, ax = plt.subplots()
    sns.histplot(df["Sum of Life Expectancy  (both sexes)"], kde=True, ax=ax)
    ax.set_title("Distribution of Life Expectancy (both sexes)")
    plt.savefig('statistical_plot.png')
    plt.show()
    return


def statistical_analysis(df, col: str):
    """Perform statistical analysis"""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Preprocess the data: remove spaces and rename columns."""
    df.columns = df.columns.str.strip()
    return df


def writing(moments, col):
    """Print the statistical moments of a given column."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    if moments[2] > 0:
        skewness = "right skewed"
    elif moments[2] < 0:
        skewness = "left skewed"
    else:
        skewness = "not skewed"

    if moments[3] > 0:
        kurtosis_type = "leptokurtic"
    elif moments[3] < 0:
        kurtosis_type = "platykurtic"
    else:
        kurtosis_type = "mesokurtic"

    print(f'The data was {skewness} and {kurtosis_type}.')
    return


def perform_clustering(df, col1, col2):
    """Perform clustering and plot elbow method."""
    data = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    def plot_elbow_method():
        """Plot the elbow method to choose the best k."""
        inertia = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), inertia, marker='o')
        ax.set_title("Elbow Method for Optimal K")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Inertia")
        plt.savefig('elbow_plot.png')
        plt.show()
        return

    def one_silhouette_inertia():
        """Perform clustering with 3 clusters (default)."""
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        return labels, kmeans.cluster_centers_

    labels, cluster_centers = one_silhouette_inertia()
    plot_elbow_method()

    xkmeans, ykmeans = cluster_centers[:, 0], cluster_centers[:, 1]
    return labels, scaled_data, xkmeans, ykmeans, labels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot clustered data with cluster centers."""
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    ax.scatter(xkmeans, ykmeans, c='red', marker='X', s=100, label='Centroids')
    ax.set_title("K-Means Clustering")
    ax.legend()
    plt.savefig('clustering.png')
    plt.show()
    return


def perform_fitting(df, col1, col2):
    """Perform linear regression and fit model."""
    data = df[[col1, col2]].dropna()
    X = data[[col1]].values
    y = data[col2].values

    model = LinearRegression()
    model.fit(X, y)
    x_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_pred)
    return data, x_pred, y_pred


def plot_fitted_data(data, x, y):
    """Plot linear regression fitting data."""
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], color='blue', label='Data')
    ax.plot(x, y, color='red', label='Fitted Line')
    ax.set_title("Fitting with Linear Regression")
    ax.legend()
    plt.savefig('fitting.png')
    plt.show()
    return


def main():
    """Main function to call all processes."""
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Sum of Life Expectancy  (both sexes)'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    clustering_results = perform_clustering(df, 
                                            'Sum of Males  Life Expectancy', 
                                            'Sum of Females  Life Expectancy'
                                           )
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, 
                                      'Sum of Males  Life Expectancy', 
                                      'Sum of Life Expectancy  (both sexes)'
                                     )
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
