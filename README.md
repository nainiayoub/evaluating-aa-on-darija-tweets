<!-- <img src="./images/logo-csum6p.png" width="30%"> -->

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

| Author | \#Tweets | Avg \#letters/tweet | Avg \#word/tweet | \#Arabic words | \#Latin words | \%Latin words |
|--------|----------|---------------------|------------------|----------------|---------------|---------------|
| A-1    | 3249     | 42.9125             | 10.4589          | 33647          | 334           | 0.9829\%      |
| A-2    | 3249     | 26.9175             | 6.2317           | 20036          | 211           | 1.0421\%      |
| A-3    | 3240     | 19.7441             | 4.7762           | 9021           | 6454          | 41.7059\%     |
| A-4    | 3238     | 38.8644             | 9.6565           | 31081          | 187           | 0.5980\%      |
| A-5    | 3234     | 17.2189             | 5.0940           | 16417          | 57            | 0.3460\%      |
| A-6    | 3216     | 37.1054             | 8.7733           | 27873          | 342           | 1.2121\%      |
| A-7    | 3208     | 40.8668             | 8.9329           | 28592          | 65            | 0.2268\%      |
| A-8    | 3178     | 17.3187             | 4.0135           | 12751          | 4             | 0.0313\%      |
| A-9    | 3166     | 35.0123             | 8.6468           | 27251          | 125           | 0.4566\%      |
| A-10   | 3141     | 27.7956             | 6.4406           | 20205          | 25            | 0.1235\%      |
| A-11   | 3091     | 45.6324             | 10.7770          | 33114          | 198           | 0.5943\%      |
| A-12   | 3074     | 41.5074             | 10.0013          | 30725          | 19            | 0.0618\%      |
| A-13   | 3018     | 60.1736             | 13.6063          | 39610          | 1454          | 3.5408\%      |
| A-14   | 3017     | 36.4998             | 8.7268           | 19384          | 6945          | 26.3777\%     |
| A-15   | 2992     | 19.2844             | 4.8556           | 13903          | 625           | 4.3020\%      |
| A-16   | 2991     | 38.3794             | 9.2490           | 25851          | 1813          | 6.5536\%      |
| A-17   | 2974     | 22.3876             | 5.7679           | 17112          | 42            | 0.2448\%      |
| A-18   | 2909     | 29.2169             | 6.9374           | 17644          | 2537          | 12.5712\%     |
| A-19   | 2884     | 23.5291             | 6.0395           | 12074          | 5344          | 30.6809\%     |
| A-20   | 2875     | 53.1749             | 12.9280          | 36901          | 267           | 0.7183\%      |
| A-21   | 2855     | 32.6171             | 7.6070           | 21496          | 222           | 1.0221\%      |
| A-22   | 2839     | 34.6174             | 8.2595           | 23284          | 165           | 0.7036\%      |
| A-23   | 2807     | 20.3170             | 4.7955           | 13110          | 351           | 2.6075\%      |
| A-24   | 2800     | 30.3475             | 7.1907           | 20052          | 82            | 0.4072\%      |
| A-25   | 2766     | 26.5527             | 6.9577           | 18772          | 473           | 2.4577\%      |
| A-26   | 2708     | 50.2566             | 11.5439          | 31215          | 46            | 0.1471\%      |
| A-27   | 2503     | 28.9360             | 7.0755           | 17400          | 310           | 1.7504\%      |
| A-28   | 1904     | 41.9779             | 9.9038           | 18835          | 22            | 0.1166\%      |
| A-29   | 853      | 36.7901             | 7.9191           | 6752           | 3             | 0.0444\%      |
| A-30   | 186      | 60.5322             | 15.1075          | 2702           | 108           | 3.8434\%      |

