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

### Visualization

## Summary