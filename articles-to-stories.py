from update_replica_set import start_mongo_client
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from bson.objectid import ObjectId
from gridfs import GridFS
from gridfs.errors import NoFile
from sklearn import decomposition
from sklearn import cluster
import pickle

def sentences_to_list(text):
    if text is None:
        return []
    else:
        transformed_text = text.replace("."," ").replace(","," ").replace(";"," ").replace("\n"," ").replace("\r"," ")
        transformed_text = text.replace('"'," ").replace("?"," ").replace("!"," ").replace("/"," ").replace(":"," ")
        transformed_text = text.replace("("," ").replace(")"," ")
        transformed_text = transformed_text.split(" ")
        transformed_text = filter(None, transformed_text)
        return transformed_text

def get_documents(client):
    # get all the documents we're interested in from the mongo client
    # return them as a list of TaggedDocuments
    docs = {}
    # perhaps the following should all be selected only if they don't already have a cluster?
    raw_events = client.dataminr.articles.find().sort("eventTime", -1).limit(10000)
    news_events = client.raw_articles.news.find({"pubDate": {"$ne": "None"}}).sort("pubDate", -1).limit(1000)
    reuters_events = client.tr.articles.find({"newsMessage.itemSet.newsItem.itemMeta.versionCreated": {"$exists": True}}).sort("newsMessage.itemSet.newsItem.itemMeta.versionCreated", -1).limit(1000)
    production_events = client.production.articles.find().sort("pubDate", -1).limit(1000)
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
            cleaned_paragraph = ""
            for paragraph in article['newsMessage']['itemSet']['newsItem']['contentSet']['inlineXML']['html']['body']['p']:
                if paragraph is not None:
                    for i in paragraph:
                        if i is not None:
                            if ord(i) < 128:
                                cleaned_paragraph = cleaned_paragraph + i
                            else:
                                cleaned_paragraph = cleaned_paragraph + " "
            docs[str(article["_id"])] = cleaned_paragraph.replace("\n"," ")
    for production_event in production_events:
        docs[str(production_event["_id"])] = production_event["content"]
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

def update_field_in_collection(object_id,field,value,collection):
    collection.update(
        {"_id": object_id},
        {"$set": {field: value}},
        upsert = False, multi = False)

def update_field(object_id,field,value,client):
    update_field_in_collection(object_id,field,value,client.dataminr.articles)
    update_field_in_collection(object_id,field,value,client.raw_articles.articles)
    update_field_in_collection(object_id,field,value,client.tr.articles)
    update_field_in_collection(object_id,field,value,client.production.articles)

def get_field_in_collection(object_id,field_name,collection):
    try:
        value = collection.find({"_id": object_id}).limit(1)[0][field_name]
    except IndexError:
        value = None
    return value

def get_field(object_id,field_name,client):
    value = None
    value = get_field_in_collection(object_id,field_name,client.dataminr.articles)
    if value is None:
        value = get_field_in_collection(object_id,field_name,client.raw_articles.articles)
        if value is None:
            value = get_field_in_collection(object_id,field_name,client.tr.articles)
            if value is None:
                value = get_field_in_collection(object_id,field_name,client.production.articles)
    return value

def update_docvecs(docs,d2v_model,client):
    for doc in docs:
        docvec = d2v_model.infer_vector(doc.words).tolist()
        update_field(ObjectId(doc.tags[0]),"docvec",docvec,client)

def update_pcavecs(docs,pca_model,client):
    for doc in docs:
        pcavec = None
        docvec = get_field(ObjectId(doc.tags[0]),"docvec",client)
        if docvec is not None:
            try:
                pcavec = pca_model.transform(docvec)[0].tolist()
            except Exception:
                pass
            if pcavec is not None:
                update_field(ObjectId(doc.tags[0]),"pcavec",pcavec,client)

def pca_docs(docs,client,collectionname = None,filename = None):
    if collectionname is None:
        collectionname = "doc2vec"
    if filename is None:
        filename = "doc2vec_pca"
    modelstore = GridFS(client.models,collection=collectionname)
    try:
        pca_model = pickle.loads(modelstore.get_version(filename=filename).read())
    except NoFile:
        pca_model = decomposition.RandomizedPCA(n_components=3)
    training_data = []
    for doc in docs:
        try:
            doc_result = get_field(ObjectId(doc.tags[0]),"docvec",client)
            if doc_result is not None:
                training_data.append(doc_result)
        except Exception:
            pass
    pca_model.fit(training_data)
    modelstore.put(pickle.dumps(pca_model),filename=filename)
    update_pcavecs(docs,pca_model,client)

def get_vector6(object_id,client):
    timestamp = None
    lat = None
    lon = None
    pcavec1 = None
    pcavec2 = None
    pcavec3 = None
    try:
        event = client.dataminr.articles.find({"_id": object_id})[0]
        timestamp = event["displayTweet"]["eventTime"]
        lat = event["eventLocation"][0]["coordinates"][0]
        lon = event["eventLocation"][0]["coordinates"][1]
        pcavec1 = event["pcavec"][0]
        pcavec2 = event["pcavec"][1]
        pcavec3 = event["pcavec"][2]
    except IndexError:
        try:
            event = client.raw_articles.news.find({"_id": object_id})[0]
            # add code here to get the relevant data if it's a scraped article
        except IndexError:
            try:
                reuters_events = client.tr.articles.find({"_id": object_id})[0]
                # add code here to get the relevant data if it's a reuters article
            except IndexError:
                try:
                    production_events = client.production.articles.find({"_id": object_id})[0]
                    # add code here to get the relevant data if it's a production article
    return [timestamp,lat,lon,pcavec1,pcavec2,pcavec3]

def update_clusters(docs,cluster_model,client):
    return None

def cluster_docs(docs,client,collectionname = None,filename = None):
    if collectionname is None:
        collectionname = "doc2vec"
    if filename is None:
        filename = "doc2vec_clustering"
    modelstore = GridFS(client.models,collection=collectionname)
    try:
        cluster_model = pickle.loads(modelstore.get_version(filename=filename).read())
    except NoFile:
        cluster_model = cluster.Birch(n_clusters = None)
    training_data = []
    for doc in docs:
        try:
            doc_result = get_vector6(ObjectId(doc.tags[0]),client)
            if doc_result is not None:
                training_data.append(doc_result)
        except Exception:
            pass
    cluster_model.fit(training_data)
    modelstore.put(pickle.dumps(cluster_model),filename=filename)
    update_clusters(docs,cluster_model,client)

if __name__ == "__main__":
    epochs = 50
    client = start_mongo_client()
    while 1:
        docs = get_documents(client)
        model = initialize_doc2vec_model(docs)
        update_docvecs(docs,model,client)
        pca_docs(docs,client)
        cluster_docs(docs,client)
