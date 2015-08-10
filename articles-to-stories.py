from update_replica_set import start_mongo_client
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def sentences_to_list(text):
    transformed_text = text.replace("."," ").replace(","," ").replace(";"," ").replace("\n"," ").replace("\r"," ")
    transformed_text = transformed_text.split(" ")
    transformed_text = filter(transformed_text, None)
    transformed_text = filter(transformed_text, "")
    transformed_text = filter(transformed_text, " ")
    return transformed_text

def get_documents(client):
    # get all the documents we're interested in from the mongo client
    # return them as a list of TaggedDocuments
    docs = {}
    raw_events = client.dataminr.articles.find().sort("eventTime", -1).limit(10000)
    news_events = client.raw_articles.news.find({"pubDate": {"$ne": "None"}}).sort("pubDate", -1).limit(10000)
    reuters_events = client.tr.articles.find({"newsMessage.itemSet.newsItem.itemMeta.versionCreated": {"$exists": True}}).sort("newsMessage.itemSet.newsItem.itemMeta.versionCreated", -1).limit(10000)
    for event in raw_events:
        tweet_id = "tweet_" + str(event["rawHTML"]["id"])
        docs[tweet_id] = event["text"]
        if "translatedText" in event.keys():
            docs[tweet_id] = event["translatedText"]
    for news_event in news_events:
        docs[news_event["_id"]] = news_event["content"]
    for article in reuters_events:
        docs[article["_id"]] = u""
        if article['newsMessage']['itemSet']['newsItem']['contentSet']['inlineXML']['html']['body']['p'] is not None:
            for paragraph in article['newsMessage']['itemSet']['newsItem']['contentSet']['inlineXML']['html']['body']['p']:
                cleaned_paragraph = ""
                if paragraph is not None:
                    for i in paragraph:
                        if i is not None:
                            if ord(i) < 128:
                                cleaned_paragraph = cleaned_paragraph + i
                            else:
                                cleaned_paragrah = cleaned_paragraph + " "
                docs[article["_id"]] = docs[article["_id"]] + " " + str(cleaned_paragraph).replace("\n"," ")
    sentencestream = []
    for key, value in docs.iteritems():
        sentencestream.append(sentences_to_list(value))
    doclist = []
    for key, value in docs.iteritems():
        doclist.append(TaggedDocument(key,sentences_to_list(value)))
    return doclist

if __name__ == "__main__":
    epochs = 50
    client = start_mongo_client()
    while 1:
        docs = get_documents(client)
        model = Doc2Vec(docs,size=300,negative=3,min_count=1,dm=1) # set dbow_words=1 if using dbow
        model.intersect_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary = True)
        idx = 0
        while idx < epochs:
            model.train(docs)
            idx = idx + 1
        
        
        
