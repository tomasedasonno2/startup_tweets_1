# startup_tweets_1

# Before you get started:

Download these three files from Google Drive, into the project directory:
- https://drive.google.com/file/d/1SJm55q0I6rP87dzcghz63tEz_-JR3g1g/view?usp=sharing
- https://drive.google.com/file/d/1DvPVU7_Nkbfx9r57Sa9En_TeehPNx_0O/view?usp=sharing
- https://drive.google.com/file/d/1dyAtaLcsfYq9yOO-9M3zsXvN7Pn0N5nz/view?usp=sharing

They can be quite large as they are pretrained transformers for NLP (working on reducing the size)

# What this is 
This uses a few ML models along with some naive human-made NLP features to attempt to scrape a day's tweets for newly announced startups. It's built with snscrape and generates a .txt file with ≤100 URLs and tweets from 24 to 48 hours ago. 

# To use: 
Run the get_tweets.py. 
It takes around ~30 minutes to pull and run the model, with most of the time being taken up by the ML classifiers and transformers.
