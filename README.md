# Cryptocurrencies: Unsupervised Machine Learning

## Overview

Martha is a senior manager for Accountability Accounting's Advisory Services Team, and she has has asked for our help.  The investment bank is interested in entering the new world of cryptocurrency.  Bitcoin's meteoric rise over the past decade has shown the tremendous potential for profitability (although not without its risks).  However, while that high price represents what is possible when investing in crypto, it also means that jumping on board the Bitcoin train is going to require a very expensive boarding ticket.  As a result, at this stage in the virtual gold rush, it seems unlikely that you can still "strike it rich" by hopping aboard that now very packed train.

However, maybe the possibility still exists with other newer cryptocurrencies.  But, with so many different cryptocurrencies out there, how could the company hope to find good currencies to invest in when there are so many fool's gold currencies out there?  Luckily, Martha came prepared.  She has provided us with a spreadsheet featuring information on 1253 different cryptocurrencies including whether or not it's being traded, its proof type, total coins mined, and the total coin supply.  Since we don't know exactly what output we're looking for, she asks us to use unsupervised machine learning to categorize the currencies.  Hopefully, doing so will seperate the currencies with good potential from the duds and provide the company with an idea for which currencies to consider investing in.

## Results

### Preprocessing Data

While unsupervised machine learning programs are able to discover things "on their own," they first must be provided with workable data.  Our first step was to do some standard cleaning to get rid of currencies that did not meet our standards right off the bat.  We ensured that all of the currencies we considered were actively being traded using `traded_df = crypto_df[crypto_df["IsTrading"] == True]`, have a working algorithm using `working_df = traded_df[traded_df['Algorithm'].notna()]`, had complete information using `clean_df = working_df.dropna()`, and actually had coins mined with `mined_df = clean_df[clean_df["TotalCoinsMined"] > 0]`.

Once we had our baseline qualifications met, we needed to translate our non-number columns into numbers that our machine learning programs could work with.  To do so, we used the get_dummies function to instead turn those columns into different "yes (1) or no (0)" columns: `X = pd.get_dummies(mined_df, columns=['Algorithm', 'ProofType'])`.  We also used StandardScaler to scale our data so that the results wouldn't be warped by exceptionally big numbers in the data: `crypto_scaled = StandardScaler().fit_transform(X)`.

### Reducing Data Dimensions

![PCA](https://github.com/Jeffstr00/Cryptocurrencies/blob/main/Resources/pca.png)

In order to help the algorithm deal with a large number of input features, we employed the Principal Component Analysis (PCA) statistical technique to reduce the number of dimensions while still containing most of the original information.  

### Clustering Cryptocurrencies

While we originally started off with over a thousand different crytocurrencies, even after paring that number down, we were still left with a hefty list of 685 different currencies.  In order to better understand and handle that information, we next turned to clustering, which is unsupervised learning that groups similar data points together.  In this case, we used the K-means clustering algorithm, which groups data into clusters depending on their distance to a centroid point.

However, before jumping in and getting started, we had to know how many clusters the data should be divided into.  Rather than just picking a number out of the air (or resulting to trial and error), we decided that it would be prudent determine which value would actually be best.  In order to do this, we calculated the inertia that we would find for k number of clusters using the following code:
`for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(pcs_df)
    inertia.append(km.inertia_)`
Once we had the inertias calculated, we graphed it using:
`elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")`
This provided us with a graph that shows the inertia for each k number of clusters.  This is a clear drastic change (that resembles an elbow) at 4 clusters, so that is the number we will go with in order to have the most distinct groups.

![Elbow Curve](https://github.com/Jeffstr00/Cryptocurrencies/blob/main/Resources/elbow.png)

With our ideal number of clusters known, we were now able to run the K-means model to seperate the data into distinct groups:
`model = KMeans(n_clusters=4, random_state=0)
model.fit(pcs_df)
predictions = model.predict(pcs_df)`

### Visualization

While it was nice for us that we had the data separated into clusters of similar cryptocurrencies, our end goal is conveying that information to Martha and the other managers at Accountability Accounting.  While we have all of our finished information in a tidy, sortable table that is very useful, we really wanted to be able to illustrate our findings with the different coins.  In order to help accomplish this, we created both a 2-D plot displaying each coin's total supply vs. how much as been mined and an interactive 3-D scatter plot showing how the coins are clustered into four groups depending on their aformentioned principal components.

![Unsorted Table](https://github.com/Jeffstr00/Cryptocurrencies/blob/main/Resources/table_unsorted.png)

![2-D Scatter Plot](https://github.com/Jeffstr00/Cryptocurrencies/blob/main/Resources/2d_scatter.png)

![3-D Scatter Plot](https://github.com/Jeffstr00/Cryptocurrencies/blob/main/Resources/3d_scatter.png)

## Summary