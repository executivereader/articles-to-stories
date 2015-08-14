from update_replica_set import start_mongo_client
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from bson.objectid import ObjectId
from gridfs import GridFS
from gridfs.errors import NoFile
from sklearn import decomposition
import pickle

def sentences_to_list(text):
    if text is None:
        return []
    else:
        transformed_text = text.replace("."," ").replace(","," ").replace(";"," ").replace("\n"," ").replace("\r"," ")
        transformed_text = text.replace('"'," ").replace("?"," ").replace("!"," ").replace("/"," ").replace(":"," ")
        transformed_text = transformed_text.split(" ")
        transformed_text = filter(None, transformed_text)
        return transformed_text

def get_documents(client):
    # get all the documents we're interested in from the mongo client
    # return them as a list of TaggedDocuments
    docs = {}
    raw_events = client.dataminr.articles.find().sort("eventTime", -1).limit(10000)
    news_events = client.raw_articles.news.find({"pubDate": {"$ne": "None"}}).sort("pubDate", -1).limit(1000)
    reuters_events = client.tr.articles.find({"newsMessage.itemSet.newsItem.itemMeta.versionCreated": {"$exists": True}}).sort("newsMessage.itemSet.newsItem.itemMeta.versionCreated", -1).limit(1000)
    for event in raw_events:
        if "displayTweet" in event.keys():
            if "id" in event["displayTweet"].keys():
                docs[str(event["_id"])] = event["displayTweet"]["text"]
                if "translatedText" in event["displayTweet"].keys():
                    docs[str(event["_id"])] = event["displayTweet"]["translatedText"]
    for news_event in news_events:
        docs[str(news_event["_id"])] = news_event["content"]
    for article in reuters_events:
        docs[str(article["_id"])] = u""
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
                docs[str(article["_id"])] = docs[str(article["_id"])] + " " + str(cleaned_paragraph).replace("\n"," ")
    #sentencestream = [] # what is this part for?
    #for key, value in docs.iteritems():
    #    sentencestream.append(sentences_to_list(value))
    doclist = []
    for key, value in docs.iteritems():
        doclist.append(TaggedDocument(words=sentences_to_list(value),tags=[key]))
    return doclist

def initialize_doc2vec_model(docs,filename = None, intersect = None, epochs = None):
    if filename is None:
        filename = "GoogleNews-vectors-negative300.bin.gz"
    if intersect is None:
        intersect = True
    if epochs is None:
        epochs = 10
    model = Doc2Vec(docs,size=300,negative=3,min_count=1,dm=1) # set dbow_words=1 if using dbow
    if intersect:
        try:
            model.intersect_word2vec_format(filename)
        except Exception:
            pass
    idx = 0
    while idx < epochs:
        model.train(docs) # might need to worry about the training rate here
        idx = idx + 1
    return model

def update_docvecs_in_collection(doc,docvec,collection):
    collection.update(
        {"_id": ObjectId(doc.tags[0])},
        {"$set": {"docvec": docvec}},
        upsert = False, multi = False)

def update_docvecs(docs,model,client):
    for doc in docs:
        docvec = model.infer_vector(doc.words).tolist()
        update_docvecs_in_collection(doc,docvec,client.dataminr.articles)
        update_docvecs_in_collection(doc,docvec,client.raw_articles.news)
        update_docvecs_in_collection(doc,docvec,client.tr.articles)
        update_docvecs_in_collection(doc,docvec,client.production.articles)

def cluster_docs(docs,client,collectionname = None,filename = None):
    if collectionname is None:
        collectionname = "doc2vec"
    if filename is None:
        filename = "doc2vec_pca"
    modelstore = GridFS(client.models,collection=collectionname)
    try:
        pca_model = pickle.loads(modelstore.get_version(filename=filename).read())
    except NoFile:
        pca_model = decomposition.RandomizedPCA(n_components=3)
    for doc in docs:
        # need to go through all the documents and get their principal components
        # but is it best to use docs or should we re-run the query in mongo?
    # now we'll need to do the actual clustering here

if __name__ == "__main__":
    epochs = 50
    client = start_mongo_client()
    while 1:
        docs = get_documents(client)
        model = initialize_doc2vec_model(docs)
        update_docvecs(docs,model,client)
        cluster_docs(docs,client)
