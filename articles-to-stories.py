from update_replica_set import start_mongo_client
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from bson.objectid import ObjectId
from gridfs import GridFS
from gridfs.errors import NoFile
from sklearn import decomposition
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import pickle
import datetime
import time

PCAVECTORSIZE = 20

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

def iterator_to_docs(iterator):
    # get all the documents we're interested in from the mongo client
    # return them as a list of TaggedDocuments
    docs = {}
    for document in iterator:
        docs[str(document["_id"])] = document["content"]
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
    update_field_in_collection(object_id,field,value,client.production.staging)

def get_field_in_collection(object_id,field_name,collection):
    try:
        value = collection.find({"_id": object_id}).limit(1)[0][field_name]
    except IndexError, KeyError:
        value = None
    return value

def get_field(object_id,field_name,client):
    value = None
    value = get_field_in_collection(object_id,field_name,client.production.staging)
    return value

def update_docvecs(docs,d2v_model,client):
    for doc in docs:
        docvec = d2v_model.infer_vector(doc.words).tolist()
        update_field(ObjectId(doc.tags[0]),"docvec",docvec,client)

def get_vector(object_id,client):
    timestamp = None
    lat = None
    lon = None
    try:
        event = client.production.staging.find({"_id": object_id})[0]
    except IndexError:
        return None
    try:
        epoch = datetime.datetime.utcfromtimestamp(0)
        delta = event["dateProcessed_ER"] - epoch # this may not be the best timestamp to use but meh
        timestamp = delta.total_seconds()
    except Exception:
        return None
    try:
        geos = event["geos"]
    except KeyError:
        return None
    if len(geos) == 0:
        return None
    while isinstance(geos[0],list): # get the first coordinate pair in a list of lists
        geos = geos[0]
    if len(geos) < 2:
        return None
    if isinstance(geos[1],float):
        lat = geos[1]
    else:
        return None
    if isinstance(geos[0],float):
        lon = geos[0]
    else:
        return None
    returnlist = [timestamp,lat,lon]
    for docvec_element in event["docvec"]:
        try:
            returnlist.append(docvec_element)
        except Exception:
            return None
    else:
        return returnlist

def update_pcavecs(docs,pca_model,scaler,client):
    for doc in docs:
        pcavec = None
        vector = get_vector(ObjectId(doc.tags[0]),client)
        if vector is not None:
            try:
                pcavec = pca_model.transform(scaler.transform(vector))[0].tolist()
            except Exception:
                pass
            if pcavec is not None:
                update_field(ObjectId(doc.tags[0]),"pcavec",pcavec,client)

def pca_docs(docs,client,collectionname = None,filename = None,scalerfilename = None):
    if collectionname is None:
        collectionname = "doc2vec"
    if filename is None:
        filename = "doc2vec_pca"
    if scalerfilename is None:
        scalerfilename = "doc2vec_pca_scaler"
    modelstore = GridFS(client.models,collection=collectionname)
    try:
        pca_model = pickle.loads(modelstore.get_version(filename=filename).read())
    except NoFile:
        pca_model = decomposition.RandomizedPCA(n_components=PCAVECTORSIZE)
    if pca_model.n_components != PCAVECTORSIZE:
        pca_model = decomposition.RandomizedPCA(n_components=PCAVECTORSIZE)
    training_data = []
    for doc in docs:
        try:
            doc_result = get_vector(ObjectId(doc.tags[0]),client)
            if doc_result is not None:
                training_data.append(doc_result)
        except Exception:
            pass
    try:
        scaler = pickle.loads(modelstore.get_version(filename=scalerfilename).read())
    except NoFile:
        scaler = StandardScaler()
    scaler.fit(training_data)
    pca_model.fit(scaler.transform(training_data))
    modelstore.put(pickle.dumps(pca_model),filename=filename)
    modelstore.put(pickle.dumps(scaler),filename=scalerfilename)
    update_pcavecs(docs,pca_model,scaler,client)

def update_clusters(docs,cluster_model,client):
    for doc in docs:
        print doc.tags[0]
        vector = get_field(ObjectId(doc.tags[0]),"pcavec",client)
        if vector is not None:
            if len(vector) == PCAVECTORSIZE:
                prediction = cluster_model.predict(vector).tolist()
                update_field(ObjectId(doc.tags[0]),"story",prediction,client)

def cluster_docs(docs,client,collectionname = None,filename = None):
    if collectionname is None:
        collectionname = "doc2vec"
    if filename is None:
        filename = "doc2vec_clustering"
    modelstore = GridFS(client.models,collection=collectionname)
    try:
        cluster_model = pickle.loads(modelstore.get_version(filename=filename).read())
    except NoFile:
        cluster_model = cluster.DBSCAN(eps=1.0,min_samples=1)
    training_data = []
    for doc in docs:
        try:
            doc_result = get_field(ObjectId(doc.tags[0]),"pcavec",client)
            if doc_result is not None:
                training_data.append(doc_result)
                print doc_result
        except Exception:
            print "Could not get vector"
            pass
    cluster_model.fit(training_data)
    modelstore.put(pickle.dumps(cluster_model),filename=filename)
    update_clusters(docs,cluster_model,client)

if __name__ == "__main__":
    epochs = 50
    print "Initializing Mongo client"
    client = start_mongo_client()
    print "Initializing doc2vec model"
    docs = iterator_to_docs(client.production.staging.find({}).sort("dateProcessed_ER", -1).limit(5000))
    model = initialize_doc2vec_model(docs)
    while 1:
        try:
            print "Updating document vectors"
            docs = iterator_to_docs(client.production.staging.find({"docvec": {"$exists": False}}).sort("dateProcessed_ER", -1).limit(5000))
            update_docvecs(docs,model,client)
        except Exception:
            print "Did not update document vectors"
        try:
            print "Updating PCA vectors"
            docs = iterator_to_docs(client.production.staging.find({"pcavec": {"$exists": False}, "docvec": {"$exists": True}}).sort("dateProcessed_ER", -1).limit(5000))
            pca_docs(docs,client)
        except Exception:
            print "Did not update PCA vectors"
        try:
            print "Updating clusters"
            docs = iterator_to_docs(client.production.staging.find({"pcavec": {"$exists": True}}).sort("dateProcessed_ER", -1).limit(5000))
            cluster_docs(docs,client)
        except Exception:
            print "Did not update clusters"
        time.sleep(5)
