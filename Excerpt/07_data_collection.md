# Data collection

Data was collected from [dev.to][dev-dot-to], a platform where developers collaborate and write articles, using the the exposed restful api. Head over to [dev.to][dev-dot-to] to login and to the [extensions][api-key-gen] to generate an API key. Review the restful api at, [forem developers][api-integration-dev-dot-to]. Below are the steps that demonstrates how to pull articles and comments to from [dev.to][dev-dot-to]:

- create a `.env` file in the project's root folder
- add the variable, `API_KEY`, and assign it the api key generated previously
- read the url path of the articles into a separate file, `article_paths.txt` by running, `python data_collection_read_url_paths.py`
- after running `python data_collection_read_articles_by_url_path.py` and on success, there will be a new file, `dev_articles.csv` which will have the response from the api request (unprocessed).
- process the raw data by running, `python data_collection_preprocess_read_articles.py` which will create `preprocess_dev_articles.csv`

#

[dev-dot-to]: dev.to
[about-dev-dot-to]: https://dev.to/about
[api-key-gen]: https://dev.to/settings/extensions
[api-integration-dev-dot-to]: https://developers.forem.com/api/v1
