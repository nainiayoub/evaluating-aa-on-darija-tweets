<img src="./images/logo-csum6p.png" width="30%">

# Evaluating Author Attribution on Darija Tweets

__Table of contents__
* [Abstract](#abstract)
* [Data Collection](#data-collection)
  * [Obtaining a set of Moroccan Twitter accounts](#obtaining-a-set-of-moroccan-twitter-accounts)
  * [Dataset statistics](#statistics-of-the-moroccan-darija-tweets-aa-dataset)

### Abstract
This work aims to apply and assess the existing Authorship Attribution techniques, on Moroccan Arabic (Darija) social media electronic texts (tweets). Authorship Attribution is a stylometry problem that aims to deduce the identity of the authors by examining e-texts only. 
We introduce the Moroccan Darija Author Attribution Tweets datasets with 30 authors in total so far [`dataset_directory = './data/'`](https://github.com/nainiayoub/evaluating-aa-on-darija-tweets/tree/main/data), as well as our approach to build a Random Forests Author Attribution model.

### Data Collection
#### Obtaining a set of Moroccan Twitter Accounts
The initial proposed approach wass to extract Twitter trends for Morocco using Twitter API via the WOEID (Where On Earth ID) of the country. We used the [WOEID Search Engine](https://www.woeids.com/) to look for the WOEID of Morocco _(23424893)_ and then place it as an argument to the Twitter Trends API. However, the Trends API does not cover Morocco. Hence, we proceeded with the following approach:

| Algorithm 1: Obtaining a set of Moroccan Twitter Accounts                                                          |
|--------------------------------------------------------------------------------------------------------------------|
|1. Manually identifying Twitter trends for Morocco.                                                                 |
|2. Extracttweets for each trend (sorted by Followers Count and Total Tweets).                                       |
|3. Verify the type of tweets for the top tweet authors extracted in 2.                                              | 
|4. Save the authors username if the tweets are valid (in Moroccan Darija and total number of tweets is important).  |                        


| Algorithm 2 Downloading and Preprocessing Tweets                                                 |
|--------------------------------------------------------------------------------------------------|
|1. Save the maximum number of tweets as allowed by the Twitter application programming interface. |
|2. Drop all reposted tweets, as the owner of the account did not write them.                      |
|3. Use placeholders to replace all tags, user names, and URLs.                                    |
|4. Save author identiﬁers.                                                                        |

### Statistics of the Moroccan Darija tweets AA dataset

| Author | Chunks | Avg \#letters/chunk | Avg \#words/chunk |
|--------|--------|---------------------|-------------------|
| A-1    | 10     | 13942.3             | 3398.1            |
| A-2    | 10     | 8745.5              | 2024.7            |
| A-3    | 10     | 6397.1              | 1547.5            |
| A-4    | 10     | 12584.3             | 3126.8            |
| A-5    | 10     | 5568.6              | 1647.4            |
| A-6    | 10     | 11933.1             | 2821.5            |
| A-7    | 10     | 13110.1             | 2865.7            |
| A-8    | 10     | 5503.9              | 1275.5            |
| A-9    | 10     | 11084.9             | 2737.6            |
| A-10   | 10     | 8730.6              | 2023              |
| A-11   | 10     | 14105               | 3331.2            |
| A-12   | 10     | 12759.4             | 3074.4            |
| A-13   | 10     | 18160.4             | 4106.4            |
| A-14   | 10     | 11012               | 2632.9            |
| A-15   | 10     | 5769.9              | 1452.8            |
| A-16   | 10     | 11479.3             | 2766.4            |
| A-17   | 10     | 6658.1              | 1715.4            |
| A-18   | 10     | 8499.2              | 2018.1            |
| A-19   | 10     | 6785.8              | 1741.8            |
| A-20   | 10     | 15287.8             | 3716.8            |
| A-21   | 10     | 9312.2              | 2171.8            |
| A-22   | 10     | 9827.9              | 2344.9            |
| A-23   | 10     | 5703                | 1346.1            |
| A-24   | 10     | 8497.3              | 2013.4            |
| A-25   | 10     | 7344.5              | 1924.5            |
| A-26   | 10     | 13609.5             | 3126.1            |
| A-27   | 10     | 7242.7              | 1771              |
| A-28   | 10     | 7992.6              | 1885.7            |
| A-29   | 10     | 3138.2              | 675.5             |
| A-30   | 10     | 1125.9              | 281               |


